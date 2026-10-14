//! 流式请求编排：把上游 SSE 经「报文层 chunk 钩子 → 解码 → Core 层事件钩子 →
//! 编码 → 报文层 chunk 钩子」链路转换为客户端协议的 SSE 输出流。
//!
//! 输出采用 axum 原生 [`Sse`]，流项为 `Result<Event, Infallible>`。
//!
//! 收尾审计（usage 写入数据库 + trace 写入磁盘）由 [`StreamAudit`] 的 `Drop` 承担，
//! 见其文档——流可能以三种方式结束（正常、中途出错、客户端断开），只有 Drop
//! 能同时覆盖。

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use moonbridge_core::{CoreStreamEvent, Protocol, StreamDelta, Usage};
use moonbridge_protocol::{
    ChunkVerdict, RawBody, RawChunk, ReqCtx, StreamDecodeState, StreamEncodeState,
};
use moonbridge_store::Database;
use serde_json::Value;

use crate::state::AppState;
use crate::trace::TraceRecord;
use crate::upstream;
use crate::usage;
use crate::GatewayError;

/// 取客户端 chunk 应写入 `data:` 字段的文本。
fn chunk_data_text(cc: &RawChunk) -> String {
    match &cc.data {
        RawBody::Json { value } => value.to_string(),
        RawBody::Text { text } => text.clone(),
        // 二进制无法直接进 SSE 文本帧：尽力转 UTF-8（保留可观测性，不 panic）
        RawBody::Binary { data } => String::from_utf8_lossy(data).to_string(),
        RawBody::Empty => cc.raw.clone(),
    }
}

/// 把客户端方向的 RawChunk 转为 axum SSE [`Event`]。
fn chunk_to_event(cc: &RawChunk) -> Event {
    let mut ev = Event::default();
    if let Some(name) = &cc.event {
        ev = ev.event(name.clone());
    }
    ev.data(chunk_data_text(cc))
}

/// 构造一个带内 error 事件（流已开始、HTTP 200 已发出，错误只能带内传递）。
fn error_event(msg: &str) -> Event {
    Event::default()
        .event("error")
        .data(serde_json::json!({ "error": { "message": msg } }).to_string())
}

/// 流事件聚合器：把 Core 事件流还原成「最终响应消息」快照。
///
/// trace 的 `upstream_response` / `client_response` 在流式下用它写入磁盘：
/// 体积上界 = 最终消息体（不含逐 chunk 时序与协议帧开销）。
#[derive(Default)]
struct StreamAssembler {
    id: String,
    model: String,
    /// 按 Core 块索引聚合的内容块；None = 该索引只见增量未见实体。
    blocks: Vec<Option<moonbridge_core::ContentBlock>>,
    /// ToolUse 的 partial_json 逐块累积，finish 时一次性 parse。
    tool_args: std::collections::HashMap<usize, String>,
    stop_reason: Option<moonbridge_core::StopReason>,
    usage: Usage,
}

impl StreamAssembler {
    /// 取 `index` 处槽位，按需扩容。
    fn slot(&mut self, index: usize) -> &mut Option<moonbridge_core::ContentBlock> {
        if self.blocks.len() <= index {
            self.blocks.resize(index + 1, None);
        }
        &mut self.blocks[index]
    }

    /// 取 `index` 处内容块，缺失时建占位块（chat 系上游正文是裸 delta，无 BlockStart）。
    fn block_or(
        &mut self,
        index: usize,
        make: impl FnOnce() -> moonbridge_core::ContentBlock,
    ) -> &mut moonbridge_core::ContentBlock {
        let slot = self.slot(index);
        if slot.is_none() {
            *slot = Some(make());
        }
        slot.as_mut().expect("slot just filled")
    }

    fn reasoning_block() -> moonbridge_core::ContentBlock {
        moonbridge_core::ContentBlock::Reasoning {
            text: String::new(),
            signature: None,
            redacted: false,
        }
    }

    fn feed(&mut self, ev: &CoreStreamEvent) {
        use moonbridge_core::ContentBlock as B;
        match ev {
            CoreStreamEvent::MessageStart { id, model } => {
                self.id.clone_from(id);
                self.model.clone_from(model);
            }
            CoreStreamEvent::BlockStart { index, block } => {
                *self.slot(*index) = Some(block.clone());
            }
            CoreStreamEvent::BlockDelta { index, delta } => {
                let i = *index;
                match delta {
                    StreamDelta::Text { text } => {
                        if let B::Text { text: t } = self.block_or(i, || B::text("")) {
                            t.push_str(text);
                        }
                    }
                    StreamDelta::Reasoning { text } => {
                        if let B::Reasoning { text: t, .. } =
                            self.block_or(i, Self::reasoning_block)
                        {
                            t.push_str(text);
                        }
                    }
                    StreamDelta::ReasoningSignature { signature } => {
                        if let B::Reasoning { signature: s, .. } =
                            self.block_or(i, Self::reasoning_block)
                        {
                            *s = Some(signature.clone());
                        }
                    }
                    StreamDelta::ToolInput { partial_json } => {
                        self.tool_args.entry(i).or_default().push_str(partial_json);
                        self.block_or(i, || B::ToolUse {
                            id: String::new(),
                            item_id: None,
                            name: String::new(),
                            namespace: None,
                            input: Value::Null,
                            signature: None,
                        });
                    }
                }
            }
            CoreStreamEvent::BlockStop { index, block } => {
                // 收尾事件携带的完整块（reasoning 的 signature 还原等）优先于累积值
                if let Some(b) = block {
                    *self.slot(*index) = Some(b.clone());
                }
            }
            CoreStreamEvent::MessageDelta { stop_reason, usage } => {
                if let Some(sr) = stop_reason {
                    self.stop_reason = Some(*sr);
                }
                if let Some(u) = usage {
                    let a = &mut self.usage;
                    a.input_tokens = a.input_tokens.max(u.input_tokens);
                    a.output_tokens = a.output_tokens.max(u.output_tokens);
                    a.cache_read_tokens = a.cache_read_tokens.max(u.cache_read_tokens);
                    a.cache_write_tokens = a.cache_write_tokens.max(u.cache_write_tokens);
                    a.reasoning_tokens = a.reasoning_tokens.max(u.reasoning_tokens);
                }
            }
            CoreStreamEvent::MessageStop | CoreStreamEvent::Error { .. } => {}
        }
    }

    /// 聚合完成 → CoreResponse JSON；流上没有任何内容时返回 Null（不伪造空响应）。
    fn finish(mut self) -> Value {
        use moonbridge_core::ContentBlock as B;
        for (i, raw) in std::mem::take(&mut self.tool_args) {
            if let Some(Some(B::ToolUse { input, .. })) = self.blocks.get_mut(i) {
                *input = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
            }
        }
        if self.blocks.is_empty() && self.id.is_empty() {
            return Value::Null;
        }
        serde_json::to_value(moonbridge_core::CoreResponse {
            id: self.id,
            model: self.model,
            content: self.blocks.into_iter().flatten().collect(),
            stop_reason: self.stop_reason,
            usage: self.usage,
            ext: moonbridge_core::Map::new(),
        })
        .unwrap_or(Value::Null)
    }
}

/// 单个 Core 事件的客户端方向处理管线：流事件钩子 → 内容块过滤 → 协议编码。
/// 返回编码出的客户端 chunk（被钩子/过滤器丢弃时为空）。
/// 错误以 `Err(String)` 返回，由调用方决定如何终结。
async fn process_stream_event(
    hooks: &Arc<dyn moonbridge_protocol::PluginHooks>,
    ctx: &ReqCtx,
    client_stream: &dyn moonbridge_protocol::ClientStreamAdapter,
    enc_state: &mut StreamEncodeState,
    dropped_blocks: &mut std::collections::HashSet<usize>,
    ev: &mut CoreStreamEvent,
) -> Result<Vec<RawChunk>, String> {
    // [CORE] 流事件钩子（返回 true 丢弃）
    match hooks.on_stream_event(ctx, ev).await {
        Ok(true) => return Ok(Vec::new()),
        Err(e) => tracing::warn!(error = %e, "on_stream_event 失败，放行"),
        Ok(false) => {}
    }
    // [CORE] 内容块过滤。只有 BlockStart 携带完整块，故只对它问
    // `filter_content`；被丢的块必须连带压制同 index 的 BlockDelta/BlockStop，
    // 否则客户端会收到一个没有头（也没有尾）的孤立增量。
    match ev {
        CoreStreamEvent::BlockStart { index, block } => {
            match hooks.filter_content(ctx, block).await {
                Ok(true) => {
                    dropped_blocks.insert(*index);
                    return Ok(Vec::new());
                }
                Err(e) => tracing::warn!(error = %e, "filter_content 失败，放行"),
                Ok(false) => {}
            }
        }
        CoreStreamEvent::BlockDelta { index, .. } | CoreStreamEvent::BlockStop { index, .. }
            if dropped_blocks.contains(index) =>
        {
            return Ok(Vec::new());
        }
        _ => {}
    }
    // [CORE] 编码为客户端 SSE chunk
    client_stream
        .encode(ctx, ev, enc_state)
        .map_err(|e| e.to_string())
}

/// 流式请求的收尾审计：析构时落 usage 与 trace。
///
/// 三处历史缺陷一并修掉：
/// 1. 原先把 `usage::record` 写在 `while` 之后，客户端中途断开会 drop 掉整个
///    生成器 ⇒ 该请求既无用量也无 trace。放在 `Drop` 里则三种结束方式都覆盖。
/// 2. 原先无条件记 `status="ok"`，即使刚发完 error 事件就 break ⇒ 失败被记成成功。
///    现在按 `failed` / `completed` 记 `error` / `aborted` / `ok`。
/// 3. trace 的 status 与 error 原先从不回填（恒为构造时的 "ok"）。
struct StreamAudit {
    db: Arc<Database>,
    ctx: ReqCtx,
    upstream_model: String,
    trace_dir: Option<String>,
    /// 请求/响应体可选记录：关闭时写入磁盘前抹掉报文体。
    record_bodies: bool,
    /// trace 保留条数（0 = 不清理）。
    trace_retention: usize,
    trace: Option<TraceRecord>,
    acc: Usage,
    ttft_ms: Option<i64>,
    start: Instant,
    /// 中途失败原因；有值即整条流记为 error。
    failed: Option<String>,
    /// 上游流是否自然读尽（未读尽且无失败 = 客户端断开等提前终止）。
    completed: bool,
    /// 上游侧聚合：decode 后、Core 钩子前送入——记录上游实际发送的内容。
    up_asm: StreamAssembler,
    /// 客户端侧聚合：仅送入实际编码下发的事件——记录客户端实际收到的内容
    /// （含插件过滤、会话水印注入的效果，可与上游侧对比插件行为）。
    down_asm: StreamAssembler,
    /// 最后一个上游 chunk 的到达时刻（初始化为请求起点）。aborted 时
    /// 用它区分「上游静默超时后客户端断开」（间隔大）与「流正活跃时被掐」
    /// （间隔≈0，客户端取消/进程退出/任务中止）。
    last_chunk_at: Instant,
}

impl StreamAudit {
    /// 收尾状态：`ok` / `error` / `aborted`，以及附带的错误消息。
    fn outcome(&self) -> (&'static str, Option<String>) {
        match (&self.failed, self.completed) {
            (Some(m), _) => ("error", Some(m.clone())),
            (None, false) => (
                "aborted",
                Some("流未读尽即结束（客户端断开或上游提前关闭）".to_string()),
            ),
            (None, true) => ("ok", None),
        }
    }
}

impl Drop for StreamAudit {
    fn drop(&mut self) {
        let (status, mut err) = self.outcome();
        if status == "aborted" {
            // aborted = 生成器在读流中途被 drop（下游断开或任务被杀）；距最后
            // 上游数据的间隔是关键判别量：上游挂起后经客户端超时断开时间隔会
            // 很大，而活跃流被掐（重启/panic/用户取消）间隔接近 0。
            let idle_ms = self.last_chunk_at.elapsed().as_millis();
            let msg = format!(
                "{}；距最后上游数据 {}ms",
                err.take().unwrap_or_default(),
                idle_ms
            );
            err = Some(msg);
            tracing::warn!(
                request_id = %self.ctx.request_id,
                provider = ?self.ctx.provider_key,
                model = %self.upstream_model,
                idle_ms = idle_ms as u64,
                latency_ms = self.start.elapsed().as_millis() as u64,
                "流式请求 aborted：流未读尽，生成器被 drop（客户端断开/进程退出/任务取消）"
            );
        }
        usage::record(
            &self.db,
            &self.ctx,
            &self.upstream_model,
            &self.acc,
            status,
            err.as_deref(),
            self.start,
            self.ttft_ms,
        );
        if let Some(mut t) = self.trace.take() {
            t.usage = self.acc.into();
            t.latency_ms = self.start.elapsed().as_millis() as u64;
            t.ttft_ms = self.ttft_ms.map(|v| v as u64);
            t.status = status.to_string();
            t.error = err;
            // 流式 trace 的两个响应字段由事件聚合回填（非流式在 dispatch 已填）。
            if t.stream {
                t.upstream_response = std::mem::take(&mut self.up_asm).finish();
                t.client_response = std::mem::take(&mut self.down_asm).finish();
            }
            if !self.record_bodies {
                crate::trace::strip_bodies(&mut t);
            }
            crate::trace::write(self.trace_dir.as_deref(), &t, self.trace_retention);
        }
    }
}

/// 流头窥视结果：响应已转进 chunk 流，窥视期间消费掉的 chunk 的解码事件
/// 与 decode 状态一并移交——此时客户端还一个字节都没收到，仍可安全重发。
pub struct StreamHead {
    /// 剩余未消费的上游 chunk 流。
    pub chunks: std::pin::Pin<Box<dyn futures::Stream<Item = crate::Result<RawChunk>> + Send>>,
    /// 窥视期间产出的解码事件（前导帧），移交后须按序回放。
    pub preamble: Vec<CoreStreamEvent>,
    /// 窥视用掉的 decode 状态——剩余 chunk 必须沿同一状态续解。
    pub dec_state: StreamDecodeState,
    /// 上游响应头（窥视结束即丢失 resp，留给插件错误判定用）。
    pub resp_headers: Vec<(String, String)>,
    /// 窥视是否以 EOF 收尾（true = 流已完结，chunks 恒空）。
    pub ended: bool,
}

impl StreamHead {
    /// 未窥视的流头（无流式 Adapter 时的降级构造）：原样接管响应，
    /// `build_stream_response` 内部走「缺少流式 Adapter」错误收尾。
    pub fn unpeeked(resp: reqwest::Response, protocol: Protocol, provider: Option<String>) -> Self {
        Self {
            chunks: Box::pin(upstream::response_to_chunks(resp, protocol, provider)),
            preamble: Vec::new(),
            dec_state: StreamDecodeState::default(),
            resp_headers: Vec::new(),
            ended: false,
        }
    }
}

impl StreamHead {
    /// 上游在零内容产出时宣告失败/静默终结 → 返回错误描述，整个请求可安全重发。
    /// 只要产出过任意内容块（Block*），事件已构成客户端可见语义，不可重试。
    pub fn retryable_failure(&self) -> Option<String> {
        let has_content = self.preamble.iter().any(|e| {
            matches!(
                e,
                CoreStreamEvent::BlockStart { .. }
                    | CoreStreamEvent::BlockDelta { .. }
                    | CoreStreamEvent::BlockStop { .. }
            )
        });
        if has_content {
            return None;
        }
        if let Some(m) = self.preamble.iter().find_map(|e| match e {
            CoreStreamEvent::Error { message } => Some(message.clone()),
            _ => None,
        }) {
            return Some(m);
        }
        let clean_end = self.preamble.iter().any(|e| {
            matches!(
                e,
                CoreStreamEvent::MessageStop
                    | CoreStreamEvent::MessageDelta {
                        stop_reason: Some(_),
                        ..
                    }
            )
        });
        if self.ended && !clean_end {
            return Some("上游流未产出内容即结束".to_string());
        }
        None
    }
}

/// 窥视窗口总时长：窥视期间客户端收不到任何字节——为等错误帧而无限等待会把
/// 慢思考模型（首 token 可能 30s+）的首字节时间拖垮。窗口内没等到失败/内容
/// 就照常承诺本流（前导帧回放后客户端看到的就是正常流式时序）。
const PEEK_WINDOW_SECS: u64 = 10;

/// 窥视上游流头：HTTP 200 之后、内容事件到达之前，上游仍可能以带内错误帧
/// （`response.failed` 等）宣告失败——这种「零产出即失败」形态可安全重试，
/// 由 dispatch 交给插件钩子定夺。逐 chunk 解码直到以下任一条件：
/// 首个内容块事件出现（已承诺，正常接管）/ 流内错误帧 / 终帧 / EOF /
/// 窥视窗口耗尽（照常接管）/ 出错。
/// 返回 Err = 传输/解码失败（同样视作零产出失败，可重试）。
pub async fn peek_upstream_head(
    resp: reqwest::Response,
    adapter: &dyn moonbridge_protocol::ProviderStreamAdapter,
    ctx: &ReqCtx,
    upstream_protocol: Protocol,
    provider: Option<String>,
    idle_secs: u64,
) -> crate::Result<StreamHead> {
    let resp_headers: Vec<(String, String)> = resp
        .headers()
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    let mut chunks = Box::pin(upstream::response_to_chunks(
        resp,
        upstream_protocol,
        provider,
    ));
    let mut dec_state = StreamDecodeState::default();
    let mut preamble = Vec::new();
    let mut ended = false;
    let deadline = Instant::now() + std::time::Duration::from_secs(PEEK_WINDOW_SECS);
    loop {
        // 单次等待取「相邻事件空闲上限」与「窥视窗口剩余」的较小值；
        // 窗口耗尽不算失败——照常承诺本流。
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        let wait = if idle_secs > 0 {
            remaining.min(std::time::Duration::from_secs(idle_secs))
        } else {
            remaining
        };
        let next = match tokio::time::timeout(wait, chunks.next()).await {
            Ok(n) => n,
            Err(_) => {
                if wait == remaining {
                    break; // 窗口耗尽：承诺本流
                }
                return Err(GatewayError::Other(format!(
                    "上游流头等待 {idle_secs}s 无数据（疑似挂起）"
                )));
            }
        };
        let chunk = match next {
            None => {
                ended = true;
                break;
            }
            Some(Err(e)) => return Err(e),
            Some(Ok(c)) => c,
        };
        let evs = adapter
            .decode(ctx, &mut dec_state, &chunk)
            .map_err(|e| GatewayError::Other(format!("上游流头解码失败: {e}")))?;
        let hit_content = evs.iter().any(|e| {
            matches!(
                e,
                CoreStreamEvent::BlockStart { .. }
                    | CoreStreamEvent::BlockDelta { .. }
                    | CoreStreamEvent::BlockStop { .. }
            )
        });
        let hit_terminal = evs.iter().any(|e| {
            matches!(
                e,
                CoreStreamEvent::Error { .. }
                    | CoreStreamEvent::MessageStop
                    | CoreStreamEvent::MessageDelta {
                        stop_reason: Some(_),
                        ..
                    }
            )
        });
        preamble.extend(evs);
        if hit_content || hit_terminal {
            break;
        }
    }
    Ok(StreamHead {
        chunks,
        preamble,
        dec_state,
        resp_headers,
        ended,
    })
}

/// 构建流式响应。`head` 为 `peek_upstream_head` 窥视过的上游流头。
//
// `audit.failed` / `audit.completed` 在本函数的生成器内写入、仅由
// `StreamAudit::Drop` 读取。rustc 的 `unused_assignments` 只做字段级数据流分析、
// 不把 Drop glue 算作一次读取，故在此误报「assigned value is never read」。
// `e2e_stream_responses_to_anthropic` 断言正常读尽的流写入数据库为 status="ok"
// （而非 Drop 默认的 "aborted"），实证这些写入确实到达了 Drop。
#[allow(unused_assignments)]
pub fn build_stream_response(
    state: Arc<AppState>,
    ctx: ReqCtx,
    head: StreamHead,
    upstream_model: String,
    start: Instant,
    trace: TraceRecord,
    session_tag: Option<String>,
) -> Response {
    let hooks = state.hooks.clone();
    let upstream_protocol = ctx.upstream_protocol.unwrap_or(Protocol::Anthropic);
    let provider_stream = state.registry.provider_stream(upstream_protocol).cloned();
    let client_stream = state.registry.client_stream(ctx.client_protocol).cloned();

    let mut audit = StreamAudit {
        db: state.db.clone(),
        ctx: ctx.clone(),
        upstream_model,
        trace_dir: state.config.trace_dir.clone(),
        record_bodies: state.config.trace_record_bodies,
        trace_retention: state.config.trace_retention,
        trace: Some(trace),
        acc: Usage::default(),
        ttft_ms: None,
        start,
        failed: None,
        completed: false,
        up_asm: StreamAssembler::default(),
        down_asm: StreamAssembler::default(),
        last_chunk_at: start,
    };

    let body = async_stream::stream! {
        let (Some(provider_stream), Some(client_stream)) = (provider_stream, client_stream) else {
            audit.failed = Some("缺少流式 Adapter".to_string());
            yield Ok::<Event, Infallible>(error_event("缺少流式 Adapter"));
            return;
        };
        let mut chunks = head.chunks;
        // 窥视期已解码出的前导事件：先于此回放，耗尽后回到上游 chunk 流。
        let mut preamble: std::collections::VecDeque<CoreStreamEvent> =
            head.preamble.into_iter().collect();
        // `filter_content` 丢弃的块序号：其后续增量一律压制（见下方 [CORE] 内容块过滤）。
        let mut dropped_blocks: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        // 会话水印状态：marker 嵌进第一个可承载推理块的明文首部（同 index 的
        // 推理增量）。凭据先行 / redacted 的推理块不可注入——入口编码器会把
        // 凭据先行的块开成 redacted_thinking 完整块，再补明文增量会扰乱
        // 客户端块状态机。整流无可用推理块则不打标记（纯 CoT 方案）。
        let mut tagged = session_tag.is_none();
        // 打标记落到的块 index：其 BlockStop 携带的完整块（chat 上游的 chat:
        // 自凭据、responses 的 reasoning item 组装）也要补上 marker——
        // 只存 done item 的客户端靠它把水印带进 transcript。
        let mut marker_idx: Option<usize> = None;
        let mut ineligible_reasoning: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        // encode 每流状态（anthropic 入口的推理块惰性开块依赖它）
        let mut enc_state = StreamEncodeState::default();
        // decode 每流状态：窥视期已用掉一部分，沿同一状态续解剩余 chunk
        let mut dec_state = head.dec_state;
        // 终结状态：上游协议（Chat 的 [DONE]、Gemini 的 finishReason）可能
        // 不发任何终帧就结束——客户端协议状态机需要一帧 MessageStop 作回退。
        let mut saw_stop = false;
        // 是否已见过「真正的收尾 delta」（stop_reason 非空）。usage-only 的
        // MessageDelta 不算——那种帧之后客户端仍等不到 finish_reason。
        let mut saw_finish = false;
        let mut saw_start = false;
        // 上游带内错误帧（协议级 error event，非 decode 失败）：流被上游宣告
        // 失败——回退收尾只补 MessageStop，不再伪造 completed 组装帧；
        // 上游携带的错误原文存下来，审计时用它而不是笼统的「以错误帧结束」。
        let mut saw_err = false;
        let mut upstream_err_msg: Option<String> = None;
        // 上游是否产出过任何解码后的事件：区分「空流」与「有内容但截断」。
        let mut saw_any = false;
        // 相邻上游事件的空闲上限（0 = 无限等待）：中转挂起连接但停发数据时，
        // `chunks.next()` 无限 pend 会让整条流僵到客户端先断开，审计只剩
        // 「客户端断开」的假象；超时按上游错误收尾。
        let idle_secs = state.config.stream_idle_timeout_secs;

        // 'stream 标签：解码/编码任一环节出错都要**真正终止**整条流。
        // 编码错误分支原先的 `break` 落在内层 `for ev in events` 里，只放弃了
        // 当前 chunk 的剩余事件，外层 while 继续消费上游并反复刷 error 事件。
        'stream: loop {
            let mut events = if let Some(ev) = preamble.pop_front() {
                // 前导事件在窥视期就已到达——首事件时刻即 ttft
                if audit.ttft_ms.is_none() {
                    audit.ttft_ms = Some(start.elapsed().as_millis() as i64);
                }
                audit.last_chunk_at = Instant::now();
                vec![ev]
            } else {
                let next = if idle_secs > 0 {
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(idle_secs),
                        chunks.next(),
                    )
                    .await
                    {
                        Ok(n) => n,
                        Err(_) => {
                            let msg = format!("上游流空闲 {idle_secs}s 无数据（疑似挂起）");
                            audit.failed = Some(msg.clone());
                            for cc in client_stream
                                .encode(
                                    &ctx,
                                    &CoreStreamEvent::Error { message: msg.clone() },
                                    &mut enc_state,
                                )
                                .unwrap_or_default()
                            {
                                yield Ok(chunk_to_event(&cc));
                            }
                            if !saw_stop {
                                for cc in client_stream
                                    .encode(&ctx, &CoreStreamEvent::MessageStop, &mut enc_state)
                                    .unwrap_or_default()
                                {
                                    yield Ok(chunk_to_event(&cc));
                                }
                            }
                            break 'stream;
                        }
                    }
                } else {
                    chunks.next().await
                };
                let Some(item) = next else { break 'stream };
                audit.last_chunk_at = Instant::now();
                let mut chunk = match item {
                    Ok(c) => c,
                    Err(e) => {
                        let msg = e.to_string();
                        audit.failed = Some(msg.clone());
                        // 带内错误走客户端协议的 Error 编码（而非裸 error 帧），
                        // 并补终结帧让客户端状态机正常收尾。
                        for cc in client_stream
                            .encode(&ctx, &CoreStreamEvent::Error { message: msg.clone() }, &mut enc_state)
                            .unwrap_or_default()
                        {
                            yield Ok(chunk_to_event(&cc));
                        }
                        if !saw_stop {
                            for cc in client_stream
                                .encode(&ctx, &CoreStreamEvent::MessageStop, &mut enc_state)
                                .unwrap_or_default()
                            {
                                yield Ok(chunk_to_event(&cc));
                            }
                        }
                        break 'stream;
                    }
                };
                if audit.ttft_ms.is_none() {
                    audit.ttft_ms = Some(start.elapsed().as_millis() as i64);
                }

                // [RAW] 上游 chunk 钩子（可改写/丢弃）
                match hooks.on_upstream_chunk_raw(&ctx, &mut chunk).await {
                    Ok(ChunkVerdict::Drop) => continue,
                    Err(e) => tracing::warn!(error = %e, "on_upstream_chunk_raw 失败，放行"),
                    Ok(ChunkVerdict::Forward) => {}
                }

                // [CORE] 解码为 Core 流事件
                match provider_stream.decode(&ctx, &mut dec_state, &chunk) {
                    Ok(ev) => ev,
                    Err(e) => {
                        let msg = e.to_string();
                        audit.failed = Some(msg.clone());
                        for cc in client_stream
                            .encode(&ctx, &CoreStreamEvent::Error { message: msg.clone() }, &mut enc_state)
                            .unwrap_or_default()
                        {
                            yield Ok(chunk_to_event(&cc));
                        }
                        if !saw_stop {
                            for cc in client_stream
                                .encode(&ctx, &CoreStreamEvent::MessageStop, &mut enc_state)
                                .unwrap_or_default()
                            {
                                yield Ok(chunk_to_event(&cc));
                            }
                        }
                        break 'stream;
                    }
                }
            };

            // 上游侧聚合：decode 后即喂（先于水印注入与 Core 钩子），
            // 记录上游实际发送的内容。
            for ev in &events {
                audit.up_asm.feed(ev);
            }

            // 先按本批事件更新流状态，再决定注入。
            for ev in &events {
                saw_any = true;
                match ev {
                    CoreStreamEvent::MessageDelta { stop_reason: Some(_), .. } => {
                        saw_finish = true;
                    }
                    CoreStreamEvent::MessageStop => saw_stop = true,
                    CoreStreamEvent::MessageStart { .. } => saw_start = true,
                    CoreStreamEvent::Error { message } => {
                        saw_err = true;
                        upstream_err_msg = Some(message.clone());
                    }
                    _ => {}
                }
            }
            // [CORE] 会话水印：向第一个可承载推理块注入首部明文增量。
            // BlockStart 到达时插在其后（块内首个增量）；chat 系上游的裸
            // 推理 delta（无 BlockStart）则插在其前，同样落在块首。
            // redacted / 凭据先行的块标记为不可注入——它们按 redacted_thinking
            // 完整块形态还原，混入明文增量会破坏客户端块结构。
            if !tagged {
                if let Some(payload) = &session_tag {
                    let mut inject: Option<(usize, usize)> = None; // (splice 位, 块 index)
                    for (i, ev) in events.iter().enumerate() {
                        match ev {
                            CoreStreamEvent::BlockStart {
                                index,
                                block: moonbridge_core::ContentBlock::Reasoning { redacted, .. },
                            } => {
                                if *redacted {
                                    ineligible_reasoning.insert(*index);
                                } else {
                                    inject = Some((i + 1, *index));
                                    break;
                                }
                            }
                            CoreStreamEvent::BlockDelta {
                                index,
                                delta: StreamDelta::ReasoningSignature { .. },
                            } => {
                                ineligible_reasoning.insert(*index);
                            }
                            CoreStreamEvent::BlockDelta {
                                index,
                                delta: StreamDelta::Reasoning { .. },
                            } if !ineligible_reasoning.contains(index) => {
                                inject = Some((i, *index));
                                break;
                            }
                            _ => {}
                        }
                    }
                    if let Some((pos, idx)) = inject {
                        events.insert(pos, crate::session::marker_reasoning_delta(payload, idx));
                        tagged = true;
                        marker_idx = Some(idx);
                    }
                }
            }

            for mut ev in events {
                // 打标记块的 BlockStop 若携带完整块（chat 上游收尾的 BlockStop
                // 带累积全文 + chat: 自凭据），其明文同样补上 marker 首部——
                // responses 入口用它组装 done item，只存完成态的客户端靠它
                // 把水印带进 transcript。凭据不补：chat: 自凭据在 decode 态
                // 早已按上游原文封口，本就只覆盖原文。
                if let (Some(mi), Some(payload)) = (marker_idx, session_tag.as_deref()) {
                    if let CoreStreamEvent::BlockStop { index, block: Some(b) } = &mut ev {
                        if *index == mi {
                            if let moonbridge_core::ContentBlock::Reasoning { text, .. } = b {
                                if !text.starts_with(crate::session::marker_head(payload).as_str()) {
                                    text.insert_str(0, &crate::session::marker_head(payload));
                                }
                            }
                        }
                    }
                }
                if let CoreStreamEvent::MessageDelta { usage: Some(u), .. } = &ev {
                    usage::accumulate(&mut audit.acc, u);
                }
                let cchunks = match process_stream_event(
                    &hooks,
                    &ctx,
                    client_stream.as_ref(),
                    &mut enc_state,
                    &mut dropped_blocks,
                    &mut ev,
                )
                .await
                {
                    Ok(c) => c,
                    Err(msg) => {
                        audit.failed = Some(msg.clone());
                        for cc in client_stream
                            .encode(&ctx, &CoreStreamEvent::Error { message: msg }, &mut enc_state)
                            .unwrap_or_default()
                        {
                            yield Ok(chunk_to_event(&cc));
                        }
                        if !saw_stop {
                            for cc in client_stream
                                .encode(&ctx, &CoreStreamEvent::MessageStop, &mut enc_state)
                                .unwrap_or_default()
                            {
                                yield Ok(chunk_to_event(&cc));
                            }
                        }
                        break 'stream;
                    }
                };
                let mut sent = false;
                for mut cc in cchunks {
                    // [RAW] 客户端 chunk 钩子（可改写/丢弃）。
                    // 出错时与上游侧对称地记 warn 后放行——原先被 `if let Ok(..)` 静默丢弃。
                    match hooks.on_client_chunk_raw(&ctx, &mut cc).await {
                        Ok(ChunkVerdict::Drop) => continue,
                        Err(e) => tracing::warn!(error = %e, "on_client_chunk_raw 失败，放行"),
                        Ok(ChunkVerdict::Forward) => {}
                    }
                    yield Ok(chunk_to_event(&cc));
                    sent = true;
                }
                // 只有真正产出了客户端字节的事件才算「客户端收到的」。
                if sent {
                    audit.down_asm.feed(&ev);
                }
            }
        }
        // 上游流自然耗尽但未产 MessageStop，分三种情形收尾：
        // 1. 已见带内错误帧（saw_err）：错误帧已是协议终态，只补 MessageStop。
        // 2. 未见 finish（!saw_finish）：上游没声明结束原因就 EOF = 异常终止——
        //    空流或截断。四种上游协议都有终帧（chat 的 finish_reason/[DONE]、
        //    responses 的 completed/incomplete、anthropic 的 message_delta+
        //    message_stop、gemini 的 finishReason），缺失即上游违约而非
        //    「协议缺终帧」。按错误收尾：已开块收口保住部分内容，发带内 Error +
        //    MessageStop，usage/trace 如实记 error——绝不能用 EndTurn 把半截
        //    回复伪造成正常完成（中转空流/断流就是这么被静默吞掉的）。
        // 3. 有 finish 但缺 MessageStop（如 chat 上游漏发 [DONE]）：内容已完整，
        //    只补客户端状态机需要的终结帧，仍记 ok。
        if !saw_stop && audit.failed.is_none() {
            let mut tail: Vec<CoreStreamEvent> = Vec::new();
            if saw_err {
                // 上游已发过带内错误帧：只补 MessageStop 让客户端状态机收尾，
                // 不再补 completed/delta——错误帧之后再说「完成」自相矛盾。
                // （审计侧按下面 saw_err 的统一标记记 error）
                tail.push(CoreStreamEvent::MessageStop);
            } else {
                if !saw_finish {
                    let msg = if saw_any {
                        "上游流未正常结束：缺少终止帧（疑似截断）".to_string()
                    } else {
                        "上游流未产出任何事件即结束".to_string()
                    };
                    audit.failed = Some(msg.clone());
                    // 有内容流出但上游没给 MessageStart 的罕见情形才补帧——
                    // 空流只发 Error + MessageStop，不伪造消息壳。
                    if !saw_start && saw_any {
                        tail.push(CoreStreamEvent::MessageStart {
                            id: String::new(),
                            model: String::new(),
                        });
                    }
                    for idx in std::mem::take(&mut dec_state.open_blocks) {
                        tail.push(CoreStreamEvent::BlockStop {
                            index: idx,
                            block: dec_state.blocks.get(&idx).cloned(),
                        });
                    }
                    tail.push(CoreStreamEvent::Error { message: msg });
                } else {
                    // 无 MessageStart 的罕见上游：先补一帧，客户端才有消息头可挂载
                    if !saw_start {
                        tail.push(CoreStreamEvent::MessageStart {
                            id: String::new(),
                            model: String::new(),
                        });
                    }
                    for idx in std::mem::take(&mut dec_state.open_blocks) {
                        tail.push(CoreStreamEvent::BlockStop {
                            index: idx,
                            block: dec_state.blocks.get(&idx).cloned(),
                        });
                    }
                }
                tail.push(CoreStreamEvent::MessageStop);
            }
            for mut ev in tail {
                // 与主循环同理：回退合成的 BlockStop 若带打标记块的完整推理块，
                // 其明文补上 marker 首部，保证 done-item 形态也含水印。
                if let (Some(mi), Some(payload)) = (marker_idx, session_tag.as_deref()) {
                    if let CoreStreamEvent::BlockStop { index, block: Some(b) } = &mut ev {
                        if *index == mi {
                            if let moonbridge_core::ContentBlock::Reasoning { text, .. } = b {
                                if !text.starts_with(crate::session::marker_head(payload).as_str()) {
                                    text.insert_str(0, &crate::session::marker_head(payload));
                                }
                            }
                        }
                    }
                }
                match process_stream_event(
                    &hooks,
                    &ctx,
                    client_stream.as_ref(),
                    &mut enc_state,
                    &mut dropped_blocks,
                    &mut ev,
                )
                .await
                {
                    Ok(cchunks) => {
                        let mut sent = false;
                        for mut cc in cchunks {
                            match hooks.on_client_chunk_raw(&ctx, &mut cc).await {
                                Ok(ChunkVerdict::Drop) => continue,
                                Err(e) => tracing::warn!(error = %e, "on_client_chunk_raw 失败，放行"),
                                Ok(ChunkVerdict::Forward) => {}
                            }
                            yield Ok(chunk_to_event(&cc));
                            sent = true;
                        }
                        if sent {
                            audit.down_asm.feed(&ev);
                        }
                    }
                    Err(e) => tracing::warn!(error = %e, "收尾事件编码失败"),
                }
            }
        }
        // 上游的带内错误帧（response.failed / anthropic error 等）是上游自己
        // 宣告失败：无论终帧齐不齐都必须记 error，不能让错误流显示为成功。
        // 标记放在收尾块之后——先让上面的分支按「还需不需要补终帧」正常发帧。
        if saw_err && audit.failed.is_none() {
            audit.failed = Some(match upstream_err_msg {
                Some(m) => format!("上游以错误帧结束流: {m}"),
                None => "上游以错误帧结束流".to_string(),
            });
        }
        if audit.failed.is_none() {
            audit.completed = true;
        }
        // usage / trace 由 `audit` 析构时统一落笔
    };

    Sse::new(body)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use moonbridge_protocol::ChunkStage;
    use serde_json::json;

    fn audit(failed: Option<&str>, completed: bool) -> StreamAudit {
        StreamAudit {
            db: Arc::new(Database::open_in_memory().unwrap()),
            ctx: ReqCtx::new("r1", Protocol::OpenAiResponse),
            upstream_model: "m-up".into(),
            trace_dir: None,
            record_bodies: true,
            trace_retention: 0,
            trace: None,
            acc: Usage::default(),
            ttft_ms: None,
            start: Instant::now(),
            failed: failed.map(str::to_string),
            completed,
            up_asm: StreamAssembler::default(),
            down_asm: StreamAssembler::default(),
            last_chunk_at: Instant::now(),
        }
    }

    /// 收尾状态机：三种流结束方式必须落到三个不同状态。
    /// 历史缺陷是无条件记 "ok"，中途失败被记成成功。
    #[test]
    fn outcome_distinguishes_ok_error_and_aborted() {
        assert!(matches!(audit(None, true).outcome(), ("ok", None)));

        let (s, e) = audit(Some("上游炸了"), false).outcome();
        assert_eq!(s, "error");
        assert_eq!(e.as_deref(), Some("上游炸了"));

        // 失败优先于「已读尽」：即使提前 break 也记 error
        let (s2, _) = audit(Some("编码失败"), true).outcome();
        assert_eq!(s2, "error");

        // 未读尽且无失败 = 客户端断开/上游提前关闭
        let (s3, e3) = audit(None, false).outcome();
        assert_eq!(s3, "aborted");
        assert!(e3.is_some(), "aborted 需带说明");
    }

    /// Drop 必须真正落一行 usage——客户端断开时生成器被 drop，
    /// 旧实现把写入数据库写在循环之后，那种情况下什么都不记。
    #[test]
    fn drop_records_usage_row() {
        let a = audit(None, false);
        let db = a.db.clone();
        assert_eq!(db.usage_summary().unwrap().requests, 0);
        drop(a);
        let sum = db.usage_summary().unwrap();
        assert_eq!(sum.requests, 1, "流被丢弃时也应留下用量记录");
        let rows = db
            .query_usage(&moonbridge_store::UsageQuery {
                limit: 5,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(rows[0].status.as_deref(), Some("aborted"));
    }

    /// 聚合器把事件流还原成 CoreResponse：裸 delta 建占位块、tool 参数
    /// 拼串后 parse、usage 取逐帧最大值。
    #[test]
    fn assembler_merges_deltas_into_response() {
        use moonbridge_core::StopReason;
        let mut a = StreamAssembler::default();
        a.feed(&CoreStreamEvent::MessageStart {
            id: "m1".into(),
            model: "gpt-x".into(),
        });
        // chat 系正文是裸 delta：无 BlockStart 也要聚出 Text 块
        a.feed(&CoreStreamEvent::BlockDelta {
            index: 0,
            delta: StreamDelta::Text { text: "Hel".into() },
        });
        a.feed(&CoreStreamEvent::BlockDelta {
            index: 0,
            delta: StreamDelta::Text { text: "lo".into() },
        });
        a.feed(&CoreStreamEvent::BlockStart {
            index: 1,
            block: moonbridge_core::ContentBlock::ToolUse {
                id: "c1".into(),
                item_id: None,
                name: "f".into(),
                namespace: None,
                input: Value::Null,
                signature: None,
            },
        });
        a.feed(&CoreStreamEvent::BlockDelta {
            index: 1,
            delta: StreamDelta::ToolInput {
                partial_json: "{\"a\":".into(),
            },
        });
        a.feed(&CoreStreamEvent::BlockDelta {
            index: 1,
            delta: StreamDelta::ToolInput {
                partial_json: "1}".into(),
            },
        });
        a.feed(&CoreStreamEvent::MessageDelta {
            stop_reason: Some(StopReason::ToolUse),
            usage: Some(Usage {
                input_tokens: 5,
                output_tokens: 3,
                ..Usage::default()
            }),
        });
        a.feed(&CoreStreamEvent::MessageStop);

        let v = a.finish();
        assert_eq!(v["id"], "m1");
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(v["content"][0]["text"], "Hello");
        assert_eq!(v["content"][1]["input"], json!({"a": 1}));
        assert_eq!(v["usage"]["input_tokens"], 5);
    }

    /// BlockStop 携带的完整块优先于累积值；空流不伪造响应。
    #[test]
    fn assembler_block_stop_wins_and_empty_is_null() {
        let mut a = StreamAssembler::default();
        a.feed(&CoreStreamEvent::BlockDelta {
            index: 0,
            delta: StreamDelta::Reasoning { text: "思".into() },
        });
        a.feed(&CoreStreamEvent::BlockStop {
            index: 0,
            block: Some(moonbridge_core::ContentBlock::Reasoning {
                text: "思考".into(),
                signature: Some("sig".into()),
                redacted: false,
            }),
        });
        let v = a.finish();
        assert_eq!(v["content"][0]["text"], "思考");
        assert_eq!(v["content"][0]["signature"], "sig");

        assert!(StreamAssembler::default().finish().is_null());
    }

    #[test]
    fn chunk_data_text_covers_every_body_kind() {
        let mk = |data: RawBody, raw: &str| RawChunk {
            stage: ChunkStage::ClientChunk,
            protocol: Protocol::OpenAiChat,
            provider: None,
            event: None,
            data,
            raw: raw.to_string(),
        };
        assert_eq!(
            chunk_data_text(&mk(RawBody::json(json!({"a": 1})), "")),
            r#"{"a":1}"#
        );
        assert_eq!(
            chunk_data_text(&mk(RawBody::text("[DONE]"), "")),
            "[DONE]",
            "Text 原样透传"
        );
        assert_eq!(
            chunk_data_text(&mk(RawBody::Empty, "fallback")),
            "fallback",
            "Empty 回落到 raw 文本"
        );
        // 二进制不得 panic
        assert!(!chunk_data_text(&mk(
            RawBody::Binary {
                data: vec![0, 255, b'a']
            },
            ""
        ))
        .is_empty());
    }
}
