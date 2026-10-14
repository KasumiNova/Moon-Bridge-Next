//! 请求生命周期编排（dispatch）。
//!
//! 严格按计划的数据流串联 RAW/CORE 两组钩子：
//! ```text
//! [RAW] on_client_request_raw → to_core_request → 路由
//!   → [CORE] on_request/inject_tools → from_core_request
//!   → [RAW] on_upstream_request_raw → 发送上游
//!   → 上游非 2xx: [RAW] on_upstream_response_raw（错误路径，msg.status >= 400）
//!                → 返回 retry 则延迟后重走上面的端点链（受 plugin_retry_max 与
//!                  plugin_retry_delay_cap_ms 钳制），其余动作按放行/短路/中止处置
//!   → 流式: read_chunks → [RAW] chunk → decode → [CORE] event → encode → [RAW] chunk → 写出
//!     非流式: [RAW] on_upstream_response_raw → to_core_response → [CORE] on_response
//!             → from_core_response → [RAW] on_client_response_raw → 回写
//! ```

use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use moonbridge_core::{CoreRequest, Protocol, Usage};
use moonbridge_protocol::{RawBody, RawMessage, RawStage, RawVerdict, ReqCtx, UpstreamRequest};
use moonbridge_store::Database;
use serde_json::Value;

use crate::error::{GatewayError, Result};
use crate::router::{ResolvedRoute, Router};
use crate::session;
use crate::state::AppState;
use crate::stream;
use crate::trace::{self, TraceRecord, TraceUsage};
use crate::upstream;
use crate::usage;

/// 插件在错误路径上要求短路代答的报文（`RawVerdict::ShortCircuit` 的三个字段）。
///
/// 单独取名有两个原因：判定要等 trace 建好之后再兑现（先取出、后应答，避免多造一份
/// trace），且元组类型本身不方便直接写在语句里。
type PluginAnswer = (u16, Vec<(String, String)>, RawBody);

/// 飞行中审计兜底：`handle_request` 的 future 挂在 axum 连接上——客户端在响应
/// 返回前断开（或进程退出/任务取消）时，整个 future 被静默 drop，`fail_audit` /
/// `finish_audit` / `StreamAudit::Drop` 一个都轮不到：usage 与 trace 零记录，
/// 上游只看到连接被掐（new-api 记 `client_gone`）。本 guard 在「正常返回」之外
/// 的一切消失路径（取消、panic 回卷）补一条 `aborted` 记录，让幽灵请求可观测。
///
/// 注意**正常返回必须先 `disarm`**——否则 Ok/Err 路径已经写过审计的会被双写；
/// 流式请求的后续生命周期由 `StreamAudit::Drop` 接管，不在本 guard 覆盖范围内。
struct InflightAudit {
    db: Arc<Database>,
    trace_dir: Option<String>,
    record_bodies: bool,
    trace_retention: usize,
    ctx: ReqCtx,
    client_request: Value,
    stream: bool,
    armed: bool,
    start: Instant,
}

impl InflightAudit {
    fn new(
        db: &Arc<Database>,
        trace_dir: Option<&str>,
        record_bodies: bool,
        trace_retention: usize,
        client_protocol: Protocol,
        raw_body: &Value,
    ) -> Self {
        let mut ctx = ReqCtx::new(uuid::Uuid::new_v4().to_string(), client_protocol);
        ctx.model_alias = raw_body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        InflightAudit {
            db: db.clone(),
            trace_dir: trace_dir.map(str::to_string),
            record_bodies,
            trace_retention,
            ctx,
            client_request: raw_body.clone(),
            stream: raw_body
                .get("stream")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            armed: true,
            start: Instant::now(),
        }
    }

    /// 正常走完的调用路径都要先解除武装（无论返回 Ok 还是 Err——Err 侧
    /// `fail_audit` 已写过 error 记录）。
    fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for InflightAudit {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let msg = "请求未完成即被取消（客户端断开、进程退出或 panic 回卷）";
        tracing::warn!(
            request_id = %self.ctx.request_id,
            model = %self.ctx.model_alias,
            client_protocol = %self.ctx.client_protocol,
            "请求在响应返回前被丢弃，上游将观测到连接中断（client_gone）"
        );
        usage::record(
            &self.db,
            &self.ctx,
            "",
            &Usage::default(),
            "aborted",
            Some(msg),
            self.start,
            None,
        );
        let mut t = new_trace(
            &self.ctx,
            self.client_request.clone(),
            self.stream,
            String::new(),
            String::new(),
            None,
            Value::Null,
        );
        t.status = "aborted".to_string();
        t.error = Some(msg.to_string());
        if !self.record_bodies {
            crate::trace::strip_bodies(&mut t);
        }
        crate::trace::write(self.trace_dir.as_deref(), &t, self.trace_retention);
    }
}

/// 处理一次入口请求，返回 axum 响应（流式 SSE 或非流式 JSON）。
pub async fn handle_request(
    state: Arc<AppState>,
    client_protocol: Protocol,
    raw_body: Value,
    req_headers: Vec<(String, String)>,
    session_id: Option<String>,
) -> Result<Response> {
    let inflight = InflightAudit::new(
        &state.db,
        state.config.trace_dir.as_deref(),
        state.config.trace_record_bodies,
        state.config.trace_retention,
        client_protocol,
        &raw_body,
    );
    let resp = run_request(state, client_protocol, raw_body, req_headers, session_id).await;
    // 返回点唯一的收口：inner 返回即正常（含 Err——错误路径已自行审计）
    inflight.disarm();
    resp
}

async fn run_request(
    state: Arc<AppState>,
    client_protocol: Protocol,
    raw_body: Value,
    req_headers: Vec<(String, String)>,
    session_id: Option<String>,
) -> Result<Response> {
    let start = Instant::now();
    let request_id = uuid::Uuid::new_v4().to_string();
    let model_alias = raw_body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let mut ctx = ReqCtx::new(request_id, client_protocol);
    ctx.session_id = session_id;
    ctx.model_alias = model_alias;

    // ── [RAW] 入站请求钩子 ──
    let mut inbound = RawMessage {
        stage: RawStage::ClientRequest,
        protocol: client_protocol,
        provider: None,
        method: Some("POST".to_string()),
        url: None,
        status: None,
        headers: req_headers,
        body: RawBody::json(raw_body),
        error: false,
        session_id: ctx.session_id.clone(),
    };
    match state
        .hooks
        .on_client_request_raw(&ctx, &mut inbound)
        .await
        .map_err(|e| {
            fail_audit(
                &state,
                &ctx,
                start,
                pre_route_trace(&ctx, &inbound),
                e.into(),
            )
        })? {
        RawVerdict::ShortCircuit {
            status,
            headers,
            body,
        } => {
            let trace = pre_route_trace(&ctx, &inbound);
            return Ok(answered(
                &state,
                &ctx,
                start,
                trace,
                Usage::default(),
                status,
                headers,
                body,
            ));
        }
        RawVerdict::Abort { message } => {
            return Err(aborted(
                &state,
                &ctx,
                start,
                pre_route_trace(&ctx, &inbound),
                message,
            ))
        }
        // 重试动作只由「上游响应钩子的错误路径」消费（见 handle_request）；
        // 其它阶段拿到它一律按放行处理。
        RawVerdict::Pass | RawVerdict::Retry { .. } => {}
    }
    // 插件会话覆写：on_client_request_raw 改写 msg.session_id（客户端自带会话
    // 头经插件转为身份源）优先于 extract_session 的结果——它是更显式的通道；
    // 置 nil 即否决宿主提取值，回退到水印/新分配路径。
    ctx.session_id = inbound.session_id.take().filter(|s| !s.is_empty());
    // 留存入站请求快照供 trace 写入磁盘（inbound.body 随后被 take_json_body 消费，
    // 失败路径也需要它构造 trace）
    let client_request_snapshot = body_snapshot(&inbound.body);
    let raw_body = match take_json_body(inbound.body, client_protocol) {
        Ok(v) => v,
        Err(e) => {
            return Err(fail_audit(
                &state,
                &ctx,
                start,
                pre_core_trace(&ctx, &client_request_snapshot),
                e,
            ))
        }
    };

    // ── 入口协议 → Core ──
    let client_adapter = state
        .registry
        .client(client_protocol)
        .ok_or_else(|| GatewayError::Route(format!("无入口 Adapter 支持协议 {client_protocol}")))
        .map_err(|e| {
            fail_audit(
                &state,
                &ctx,
                start,
                pre_core_trace(&ctx, &client_request_snapshot),
                e,
            )
        })?;
    let mut core_req = client_adapter
        .to_core_request(&ctx, raw_body)
        .await
        .map_err(|e| {
            fail_audit(
                &state,
                &ctx,
                start,
                pre_core_trace(&ctx, &client_request_snapshot),
                e.into(),
            )
        })?;
    ctx.stream = core_req.stream;

    // ── 会话水印：入站即剥除，剥完才往下走（上游与插件都看不到 marker）──
    let session_tag = resolve_session(&state, &mut ctx, &mut core_req).await;

    // ── 路由解析 ──
    let resolved = Router::resolve(&state.db, &core_req.model_alias).map_err(|e| {
        fail_audit(
            &state,
            &ctx,
            start,
            pre_core_trace(&ctx, &client_request_snapshot),
            e,
        )
    })?;
    core_req.model = resolved.upstream_model.clone();
    ctx = ctx.with_route(resolved.protocol, resolved.provider_key.clone());
    ctx.route_alias = resolved.route_alias.clone();
    ctx.upstream_model = Some(resolved.upstream_model.clone());
    ctx.upstream_max_output_tokens =
        crate::router::model_output_limit(&state.db, &resolved.upstream_model);

    // ── [CORE] 请求钩子 + 工具注入 ──
    state
        .hooks
        .on_request(&ctx, &mut core_req)
        .await
        .map_err(|e| {
            fail_audit(
                &state,
                &ctx,
                start,
                routed_trace(&ctx, &resolved, &client_request_snapshot),
                e.into(),
            )
        })?;
    let extra_tools = state.hooks.inject_tools(&ctx).await.map_err(|e| {
        fail_audit(
            &state,
            &ctx,
            start,
            routed_trace(&ctx, &resolved, &client_request_snapshot),
            e.into(),
        )
    })?;
    core_req.tools.extend(extra_tools);

    // ── Core → 上游协议（逐端点故障转移 + 插件驱动的整链重试）──

    // 外层：插件重试轮次（首轮 + 至多 plugin_retry_max 次重试）。只有上游返回错误
    // （非 2xx）且报文层钩子在错误路径上返回 {action="retry"} 才进入下一轮；重发在
    // 传输层由宿主原生完成（重走 from_core_request → 出站钩子 → send），协议翻译、
    // 流式、trace 机制全部自然复用——插件不必自己搬报文（那会绕过协议转换与流式）。
    // 内层：端点故障转移——按序尝试各端点，连接错误/超时、429、5xx 且还有后续端点时切换；
    // 其余 4xx 或末位端点失败则快速失败。协议绑定在端点上，Adapter 逐端点选取；
    // outbound 钩子随端点逐次触发（重试轮同样逐端点重触发，补头类插件是幂等 set）。
    let total = resolved.endpoints.len();
    for plugin_round in 0..=state.config.plugin_retry_max {
        let mut up: Option<UpstreamRequest> = None;
        let mut resp: Option<reqwest::Response> = None;
        let mut used_protocol = resolved.protocol;
        for (attempt, ep) in resolved.endpoints.iter().enumerate() {
            let provider_adapter = state
                .registry
                .provider(ep.protocol)
                .ok_or_else(|| {
                    GatewayError::Route(format!("无上游 Adapter 支持协议 {}", ep.protocol))
                })
                .map_err(|e| {
                    fail_audit(
                        &state,
                        &ctx,
                        start,
                        routed_trace(&ctx, &resolved, &client_request_snapshot),
                        e,
                    )
                })?;
            let mut u = provider_adapter
                .from_core_request(&ctx, &core_req, ep)
                .await
                .map_err(|e| {
                    fail_audit(
                        &state,
                        &ctx,
                        start,
                        routed_trace(&ctx, &resolved, &client_request_snapshot),
                        e.into(),
                    )
                })?;

            // ── [RAW] 出站请求钩子（每端点一次）──
            let mut outbound = RawMessage {
                stage: RawStage::UpstreamRequest,
                protocol: ep.protocol,
                provider: Some(resolved.provider_key.clone()),
                method: Some(u.method.to_string()),
                url: Some(u.url.clone()),
                status: None,
                headers: u.headers.clone(),
                body: RawBody::json(u.body.clone()),
                error: false,
                session_id: ctx.session_id.clone(),
            };
            match state
                .hooks
                .on_upstream_request_raw(&ctx, &mut outbound)
                .await
                .map_err(|e| {
                    fail_audit(
                        &state,
                        &ctx,
                        start,
                        outbound_trace(&ctx, &outbound, &resolved, &client_request_snapshot),
                        e.into(),
                    )
                })? {
                RawVerdict::ShortCircuit {
                    status,
                    headers,
                    body,
                } => {
                    return Ok(answered(
                        &state,
                        &ctx,
                        start,
                        outbound_trace(&ctx, &outbound, &resolved, &client_request_snapshot),
                        Usage::default(),
                        status,
                        headers,
                        body,
                    ));
                }
                RawVerdict::Abort { message } => {
                    return Err(aborted(
                        &state,
                        &ctx,
                        start,
                        outbound_trace(&ctx, &outbound, &resolved, &client_request_snapshot),
                        message,
                    ))
                }
                // 重试动作只由「上游响应钩子的错误路径」消费（见 handle_request）；
                // 其它阶段拿到它一律按放行处理。
                RawVerdict::Pass | RawVerdict::Retry { .. } => {}
            }
            if let Err(e) = apply_outbound(&mut u, outbound) {
                // 钩子写回的报文非法：审计按「出站钩子后的报文」无法重建（已被消费），
                // 用 Adapter 产出的原始上游请求快照。
                let t = new_trace(
                    &ctx,
                    client_request_snapshot.clone(),
                    core_req.stream,
                    resolved.upstream_model.clone(),
                    resolved.provider_key.clone(),
                    Some(ep.protocol),
                    upstream_request_snapshot(&u),
                );
                return Err(fail_audit(&state, &ctx, start, t, e));
            }

            // 非流式请求施加 request_timeout_secs 总超时（流式刻意不设——长生成
            // 不应被网关截断）。预算是整条请求的总量：端点故障转移与插件重试共享
            // 同一时钟（start），每次发送只拿剩余预算——否则累计等待会超出总额，
            // 响应体未及时就绪时，上游的 200 还会被 non_stream 的读体剩余预算改判成 504。
            // send 只覆盖到响应头；body 读取在 non_stream 里以剩余预算再套一次
            // 超时 + read_body_capped 的大小上限作回退。
            let send_res = if core_req.stream {
                upstream::send(&state.client, &u).await
            } else {
                let remaining = std::time::Duration::from_secs(state.config.request_timeout_secs)
                    .saturating_sub(start.elapsed());
                match tokio::time::timeout(remaining, upstream::send(&state.client, &u)).await {
                    Ok(r) => r,
                    Err(_) => Err(GatewayError::Upstream {
                        status: 504,
                        message: format!("上游请求超时（{}s）", state.config.request_timeout_secs),
                    }),
                }
            };
            match send_res {
                Ok(r) => {
                    let status = r.status();
                    let retryable = status.as_u16() == 429 || status.is_server_error();
                    if retryable && attempt + 1 < total {
                        // 排空响应体（有界）：连接可复用且异常上游的大 body 不会耗尽内存
                        let _ = read_body_capped(r, state.config.max_body_bytes).await;
                        tracing::warn!(
                            provider = %resolved.provider_key,
                            endpoint = %ep.base_url,
                            status = status.as_u16(),
                            "端点失败，故障转移到下一端点"
                        );
                        continue;
                    }
                    up = Some(u);
                    resp = Some(r);
                    used_protocol = ep.protocol;
                    break;
                }
                Err(e) => {
                    if attempt + 1 < total {
                        tracing::warn!(
                            provider = %resolved.provider_key,
                            endpoint = %ep.base_url,
                            error = %e,
                            "端点请求失败，故障转移到下一端点"
                        );
                        continue;
                    }
                    // 末位端点传输失败：整条链在没有任何上游响应的情况下终结——
                    // 仍须落 usage + trace，否则失败请求对用量/Traces 完全不可见。
                    let t = new_trace(
                        &ctx,
                        client_request_snapshot.clone(),
                        core_req.stream,
                        resolved.upstream_model.clone(),
                        resolved.provider_key.clone(),
                        Some(ep.protocol),
                        upstream_request_snapshot(&u),
                    );
                    return Err(fail_audit(&state, &ctx, start, t, e));
                }
            }
        }
        // 路由器保证 endpoints 非空，循环必以成功或提前 return 结束
        let up = up.expect("endpoints 非空");
        let resp = resp.expect("endpoints 非空");

        // 协议绑定在端点上：以实际命中的端点协议覆写 ctx（流式回程按它选流式 Adapter）
        ctx = ctx.with_route(used_protocol, resolved.provider_key.clone());

        let status = resp.status();

        if !status.is_success() {
            // 错误 body 可能有界很大（HTML 错误页、异常上游）——有界读取；响应头必须在
            // read_body_capped 消费 resp 之前取出，错误路径的报文钩子要用它。
            let resp_headers: Vec<(String, String)> = resp
                .headers()
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                .collect();
            let bytes = read_body_capped(resp, state.config.max_body_bytes)
                .await
                .unwrap_or_default();
            let text = String::from_utf8_lossy(&bytes).to_string();

            // 错误响应同样进报文层钩子（stage=upstream_response，body 为错误原文文本）：
            // 插件据此决定「重试 / 短路代答 / 中止 / 放行」，也可只做观测。
            // 注意此处对报文的改写**不回流客户端**——客户端拿到的是 transform_error 的
            // 产物与 gateway_error 包装；要改错误消息请用 core 层的 transform_error。
            let mut inbound_resp = RawMessage {
                stage: RawStage::UpstreamResponse,
                protocol: used_protocol,
                provider: Some(resolved.provider_key.clone()),
                method: None,
                url: None,
                status: Some(status.as_u16()),
                headers: resp_headers,
                body: RawBody::Text { text: text.clone() },
                error: true,
                session_id: ctx.session_id.clone(),
            };
            // 插件处置：None = 未要求代答（放行、或钩子出错），一律走下面的既有错误收尾
            let mut short_circuit: Option<PluginAnswer> = None;
            let mut abort_message: Option<String> = None;
            match state
                .hooks
                .on_upstream_response_raw(&ctx, &mut inbound_resp)
                .await
            {
                // 重试：延迟后重走整条端点链。配额耗尽（plugin_round 已达上限）时本臂不匹配，
                // 落到 Pass 臂按放行收尾——重试次数由宿主封顶，插件说了不算。
                Ok(RawVerdict::Retry { delay_ms })
                    if plugin_round < state.config.plugin_retry_max =>
                {
                    let delay = std::time::Duration::from_millis(
                        delay_ms.min(state.config.plugin_retry_delay_cap_ms),
                    );
                    // 非流式重试共享 request_timeout_secs 总预算：剩余预算装不下
                    // 这次等待时就放弃——睡到超时只会把真实上游错误改判成 504。
                    // 流式无总超时，不设此限。
                    let budget_ok = core_req.stream
                        || std::time::Duration::from_secs(state.config.request_timeout_secs)
                            .saturating_sub(start.elapsed())
                            > delay;
                    if budget_ok {
                        tracing::warn!(
                            provider = %resolved.provider_key,
                            status = status.as_u16(),
                            attempt = plugin_round + 1,
                            delay_ms = delay.as_millis() as u64,
                            "上游错误，插件要求重试"
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    tracing::warn!(
                        provider = %resolved.provider_key,
                        status = status.as_u16(),
                        attempt = plugin_round + 1,
                        "剩余总预算不足以完成重试，放行原始上游错误"
                    );
                }
                Ok(RawVerdict::ShortCircuit {
                    status,
                    headers,
                    body,
                }) => short_circuit = Some((status, headers, body)),
                Ok(RawVerdict::Abort { message }) => abort_message = Some(message),
                // 放行；重试配额已耗尽时也走这里（后续按 Pass 处置）
                Ok(RawVerdict::Pass) | Ok(RawVerdict::Retry { .. }) => {}
                // 钩子出错只 warn：绝不能因此掩盖原始上游错误——客户端应看到真实的上游状态码
                Err(e) => tracing::warn!(
                    provider = %resolved.provider_key,
                    status = status.as_u16(),
                    error = %e,
                    "on_upstream_response_raw 失败，按放行处理"
                ),
            }

            // ── trace 骨架（请求侧；响应侧回程时补齐）──
            // 每轮只在**终局路径**上造 trace：重试轮直接 continue（只有 warn，不落 trace），
            // 所以最终留盘的 trace 里上游请求快照是末轮重发的报文。
            // `plugin_round` 即已发生的插件重试次数（0 = 首轮，从未重试）。
            let mut trace = new_trace(
                &ctx,
                client_request_snapshot,
                core_req.stream,
                resolved.upstream_model.clone(),
                resolved.provider_key.clone(),
                Some(used_protocol),
                upstream_request_snapshot(&up),
            );
            trace.retries = plugin_round;
            if let Some((status, headers, body)) = short_circuit {
                trace.upstream_response = body_snapshot(&inbound_resp.body);
                return Ok(answered(
                    &state,
                    &ctx,
                    start,
                    trace,
                    Usage::default(),
                    status,
                    headers,
                    body,
                ));
            }
            if let Some(message) = abort_message {
                return Err(aborted(&state, &ctx, start, trace, message));
            }
            let msg = state
                .hooks
                .transform_error(&ctx, &text)
                .await
                .unwrap_or_else(|_| text.clone());
            trace.status = "error".to_string();
            // 完整响应体归位 upstream_response 快照；error 只留简短摘要，
            // 否则上游 4xx/5xx 的 HTML 错误页会整页写进 trace.error。
            trace.upstream_response = body_snapshot(&RawBody::Text { text });
            trace.error = Some(format!("上游返回 HTTP {status}"));
            finish_audit(&state, &ctx, &mut trace, start, &Usage::default(), "error");
            return Err(GatewayError::Upstream {
                status: status.as_u16(),
                message: msg,
            });
        }

        // ── 流式 / 非流式分流 ──
        // trace 骨架（请求侧；响应侧回程时补齐）；`plugin_round` 即已发生的重试次数。
        let mut trace = new_trace(
            &ctx,
            client_request_snapshot.clone(),
            core_req.stream,
            resolved.upstream_model.clone(),
            resolved.provider_key.clone(),
            Some(used_protocol),
            upstream_request_snapshot(&up),
        );
        trace.retries = plugin_round;
        return if core_req.stream {
            // ── 流头窥视：上游 200 之后、内容事件之前仍可能以带内错误帧
            // （`response.failed` 等）宣告失败——此时客户端一个字节都没收到，
            // 整个请求可安全重发。零产出失败交给插件错误钩子判定
            // （`on_upstream_response_raw` 的 error 路径；`Retry` → 重走端点链），
            // 有内容则照常接管——已产出内容的流不可重试。
            let provider_stream = state.registry.provider_stream(used_protocol);
            let head = match &provider_stream {
                Some(ps) => {
                    stream::peek_upstream_head(
                        resp,
                        ps.as_ref(),
                        &ctx,
                        used_protocol,
                        Some(resolved.provider_key.clone()),
                        state.config.stream_idle_timeout_secs,
                    )
                    .await
                }
                None => Ok(stream::StreamHead::unpeeked(
                    resp,
                    used_protocol,
                    Some(resolved.provider_key.clone()),
                )),
            };
            let retryable_err = match &head {
                Ok(h) => h.retryable_failure(),
                Err(e) => Some(e.to_string()),
            };
            if let Some(err_msg) = retryable_err {
                let mut inbound_resp = RawMessage {
                    stage: RawStage::UpstreamResponse,
                    protocol: used_protocol,
                    provider: Some(resolved.provider_key.clone()),
                    method: Some(up.method.to_string()),
                    url: Some(up.url.clone()),
                    // 上游 HTTP 层是 200——失败在流内宣告，如实记录状态码与 error 标记
                    status: Some(200),
                    headers: match &head {
                        Ok(h) => h.resp_headers.clone(),
                        Err(_) => Vec::new(),
                    },
                    body: RawBody::Text {
                        text: err_msg.clone(),
                    },
                    error: true,
                    session_id: ctx.session_id.clone(),
                };
                match state
                    .hooks
                    .on_upstream_response_raw(&ctx, &mut inbound_resp)
                    .await
                {
                    Ok(RawVerdict::Retry { delay_ms })
                        if plugin_round < state.config.plugin_retry_max =>
                    {
                        let delay = std::time::Duration::from_millis(
                            delay_ms.min(state.config.plugin_retry_delay_cap_ms),
                        );
                        tracing::warn!(
                            provider = %resolved.provider_key,
                            attempt = plugin_round + 1,
                            delay_ms = delay.as_millis() as u64,
                            error = %err_msg,
                            "上游流零产出即失败，插件要求重试"
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    Ok(RawVerdict::ShortCircuit {
                        status,
                        headers,
                        body,
                    }) => {
                        return Ok(answered(
                            &state,
                            &ctx,
                            start,
                            trace,
                            Usage::default(),
                            status,
                            headers,
                            body,
                        ));
                    }
                    Ok(RawVerdict::Abort { message }) => {
                        return Err(aborted(&state, &ctx, start, trace, message))
                    }
                    // 放行 / 重试配额耗尽：照常接管，流内错误帧如实透传客户端
                    _ => {}
                }
            }
            match head {
                Ok(head) => Ok(stream::build_stream_response(
                    state,
                    ctx,
                    head,
                    resolved.upstream_model,
                    start,
                    trace,
                    session_tag,
                )),
                Err(e) => Err(fail_audit(&state, &ctx, start, trace, e)),
            }
        } else {
            non_stream(state, ctx, resp, used_protocol, start, trace, session_tag).await
        };
    }
    // 每轮要么 continue（插件要求重试且仍有配额），要么在上面 return；末轮即使插件
    // 仍返回 retry 也按放行走错误收尾，故这里不可达。
    unreachable!("插件重试循环必然以 return 结束")
}

/// 解析本次请求归属的会话，并**就地剥除**请求里的所有 marker。
///
/// 优先级：外部显式身份（body `session_id` / `previous_response_id` /
/// `X-Codex-Window-Id`，已由 `handlers::extract_session` 填进 `ctx`）> 请求带回的
/// marker > 新分配。外部身份更可信，故两者冲突时以它为准。
///
/// marker 载荷即会话身份：uuid 载荷自带身份（活跃表淘汰 / 网关重启后带回同一
/// marker 仍落同一 `ctx.session_id`，下游亲和头如 `x-opencode-session` 不再漂移）；
/// 6-hex 短 tag 经活跃表还原外部 id，表里没有时合成确定性 `mb-{tag}`。
///
/// 返回本次响应应嵌入的 marker 载荷；水印关闭时返回 `None`（**但仍照剥 marker**——
/// 客户端可能带着开启期间留下的历史，不该让它污染上游 prompt）。
async fn resolve_session(
    state: &Arc<AppState>,
    ctx: &mut ReqCtx,
    req: &mut CoreRequest,
) -> Option<String> {
    let carried = session::extract_from_request(req);

    // 外部身份优先：无论水印开关都登记（幂等）——客户端轮换 session id 时
    // 其 `mb.session` 桶靠活跃表淘汰回收；只在水印开启时才登记会让这些桶
    // 永不被淘汰，SessionStore 无界增长。
    if let Some(id) = ctx.session_id.clone() {
        let (payload, evicted) = state.sessions.note_external(&id);
        forget_sessions(state, evicted).await;
        return state.config.session_marker.then_some(payload);
    }
    if !state.config.session_marker {
        return None;
    }
    if let Some(payload) = carried {
        let (id, evicted) = state.sessions.resolve(&payload);
        ctx.session_id = Some(id);
        forget_sessions(state, evicted).await;
        // 回发同一载荷：客户端 transcript 里的 marker 形态保持恒定
        return Some(payload);
    }
    let (id, evicted, payload) = state.sessions.new_session();
    ctx.session_id = Some(id);
    forget_sessions(state, evicted).await;
    Some(payload)
}

/// 会话被淘汰时顺手清理插件侧的会话状态（`mb.session` 的桶）。
async fn forget_sessions(state: &Arc<AppState>, evicted: Vec<String>) {
    for id in evicted {
        state.hooks.forget_session(&id).await;
    }
}

/// 非流式回程编排。`upstream_protocol` 为实际命中端点的协议；
/// `session_tag` 为本次要附加的会话水印（`None` = 不打标记）。
async fn non_stream(
    state: Arc<AppState>,
    ctx: ReqCtx,
    resp: reqwest::Response,
    upstream_protocol: Protocol,
    start: Instant,
    mut trace: TraceRecord,
    session_tag: Option<String>,
) -> Result<Response> {
    let provider_adapter = state
        .registry
        .provider(upstream_protocol)
        .ok_or_else(|| GatewayError::Route(format!("无上游 Adapter 支持协议 {upstream_protocol}")))
        .map_err(|e| fail_audit(&state, &ctx, start, trace.clone(), e))?;
    let client_adapter = state
        .registry
        .client(ctx.client_protocol)
        .ok_or_else(|| {
            GatewayError::Route(format!("无入口 Adapter 支持协议 {}", ctx.client_protocol))
        })
        .map_err(|e| fail_audit(&state, &ctx, start, trace.clone(), e))?;

    let status = resp.status().as_u16();
    let resp_headers: Vec<(String, String)> = resp
        .headers()
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    // body 读取同样受 request_timeout_secs 总预算约束（send 只覆盖响应头，
    // 慢流式上游拖 body 也要算进总超时）；超过上限或超时都以 502 报出。
    let remaining = std::time::Duration::from_secs(state.config.request_timeout_secs)
        .saturating_sub(start.elapsed());
    let bytes = match tokio::time::timeout(
        remaining,
        read_body_capped(resp, state.config.max_body_bytes),
    )
    .await
    {
        Ok(r) => r.map_err(|e| fail_audit(&state, &ctx, start, trace.clone(), e))?,
        Err(_) => {
            return Err(fail_audit(
                &state,
                &ctx,
                start,
                trace,
                GatewayError::Upstream {
                    status: 504,
                    message: format!(
                        "上游响应体读取超时（总预算 {}s）",
                        state.config.request_timeout_secs
                    ),
                },
            ))
        }
    };
    let body_value: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).to_string()));

    // ── [RAW] 入站响应钩子 ──
    let mut inbound_resp = RawMessage {
        stage: RawStage::UpstreamResponse,
        protocol: upstream_protocol,
        provider: ctx.provider_key.clone(),
        method: None,
        url: None,
        status: Some(status),
        headers: resp_headers,
        body: RawBody::json(body_value),
        error: false,
        session_id: ctx.session_id.clone(),
    };
    match state
        .hooks
        .on_upstream_response_raw(&ctx, &mut inbound_resp)
        .await
        .map_err(|e| fail_audit(&state, &ctx, start, trace.clone(), e.into()))?
    {
        RawVerdict::ShortCircuit {
            status,
            headers,
            body,
        } => {
            trace.upstream_response = body_snapshot(&inbound_resp.body);
            // 上游响应已含真实 usage（token 实际消耗、上游已计费）——被插件
            // 替换前尽力解析出来记账，否则这轮成本记 0 且账单对不上。
            let usage = match inbound_resp.body.as_json().cloned() {
                Some(v) => provider_adapter
                    .to_core_response(&ctx, v)
                    .await
                    .map(|r| r.usage)
                    .unwrap_or_default(),
                None => Usage::default(),
            };
            return Ok(answered(
                &state, &ctx, start, trace, usage, status, headers, body,
            ));
        }
        RawVerdict::Abort { message } => return Err(aborted(&state, &ctx, start, trace, message)),
        // 重试动作只由「上游响应钩子的错误路径」消费（见 handle_request）；
        // 其它阶段拿到它一律按放行处理。
        RawVerdict::Pass | RawVerdict::Retry { .. } => {}
    }
    let body_value = take_json_body(inbound_resp.body, upstream_protocol)
        .map_err(|e| fail_audit(&state, &ctx, start, trace.clone(), e))?;
    trace.upstream_response = body_value.clone();
    if !body_value.is_object() {
        // 上游 200 但 body 非 JSON 对象（HTML 错误页、纯文本等）：adapter 会把
        // 它转成全字段缺失的空响应，回给客户端一个 200 的「空成功」——必须报错。
        return Err(fail_audit(
            &state,
            &ctx,
            start,
            trace,
            GatewayError::Upstream {
                status: 502,
                message: format!(
                    "上游返回非 JSON 对象响应（HTTP {status}）：{}",
                    truncate(&body_value.to_string(), 512)
                ),
            },
        ));
    }

    // ── 上游 → Core ──
    let mut core_resp = provider_adapter
        .to_core_response(&ctx, body_value)
        .await
        .map_err(|e| fail_audit(&state, &ctx, start, trace.clone(), e.into()))?;
    // ── [CORE] 响应钩子 ──
    state
        .hooks
        .on_response(&ctx, &mut core_resp)
        .await
        .map_err(|e| fail_audit(&state, &ctx, start, trace.clone(), e.into()))?;
    // ── [CORE] 内容块过滤（响应钩子之后：插件先看全貌，再逐块决定去留）──
    if !core_resp.content.is_empty() {
        let mut kept = Vec::with_capacity(core_resp.content.len());
        for mut block in std::mem::take(&mut core_resp.content) {
            match state.hooks.filter_content(&ctx, &mut block).await {
                Ok(true) => continue,
                Ok(false) => kept.push(block),
                Err(e) => return Err(fail_audit(&state, &ctx, start, trace.clone(), e.into())),
            }
        }
        core_resp.content = kept;
    }
    // ── 会话水印：嵌进首个非 redacted 推理块的明文首部（无推理块的轮次不打标记）──
    if let Some(payload) = &session_tag {
        session::tag_response(&mut core_resp, payload);
    }
    let usage_snapshot = core_resp.usage;

    // ── Core → 入口协议 ──
    let client_json = client_adapter
        .from_core_response(&ctx, core_resp)
        .await
        .map_err(|e| fail_audit(&state, &ctx, start, trace.clone(), e.into()))?;

    // ── [RAW] 出站响应钩子 ──
    let mut outbound_resp = RawMessage {
        stage: RawStage::ClientResponse,
        protocol: ctx.client_protocol,
        provider: None,
        method: None,
        url: None,
        status: Some(200),
        headers: vec![("content-type".to_string(), "application/json".to_string())],
        body: RawBody::json(client_json),
        error: false,
        session_id: ctx.session_id.clone(),
    };
    match state
        .hooks
        .on_client_response_raw(&ctx, &mut outbound_resp)
        .await
        .map_err(|e| fail_audit(&state, &ctx, start, trace.clone(), e.into()))?
    {
        RawVerdict::ShortCircuit {
            status,
            headers,
            body,
        } => {
            return Ok(answered(
                &state,
                &ctx,
                start,
                trace,
                usage_snapshot,
                status,
                headers,
                body,
            ));
        }
        RawVerdict::Abort { message } => return Err(aborted(&state, &ctx, start, trace, message)),
        // 重试动作只由「上游响应钩子的错误路径」消费（见 handle_request）；
        // 其它阶段拿到它一律按放行处理。
        RawVerdict::Pass | RawVerdict::Retry { .. } => {}
    }
    let final_body = outbound_resp.body.as_json().cloned().unwrap_or(Value::Null);

    trace.client_response = final_body.clone();
    trace.usage = usage_snapshot.into();
    finish_audit(&state, &ctx, &mut trace, start, &usage_snapshot, "ok");

    Ok(axum::Json(final_body).into_response())
}

/// 从 RawBody 取出 JSON；文本尝试解析，其余报错。
fn take_json_body(body: RawBody, protocol: Protocol) -> Result<Value> {
    match body {
        RawBody::Json { value } => Ok(value),
        RawBody::Text { text } => serde_json::from_str(&text).map_err(|e| {
            GatewayError::Protocol(moonbridge_core::CoreError::protocol(
                protocol.to_string(),
                format!("报文改写后非法 JSON: {e}"),
            ))
        }),
        RawBody::Empty => Ok(Value::Null),
        RawBody::Binary { .. } => Err(GatewayError::Protocol(
            moonbridge_core::CoreError::protocol(
                protocol.to_string(),
                "报文为二进制，无法作为 JSON 处理",
            ),
        )),
    }
}

/// 把插件对出站报文的改写**全部**回读到 [`UpstreamRequest`]。
///
/// 历史缺陷：这里只回读 url / headers / JSON body。`method` 明明被写进了递给 Lua 的
/// 表里（`convert.rs`），却从不回读 ⇒ 插件改 `msg.method` 静默无效；插件把 body 改写成
/// `Text` 同样被 `if let RawBody::Json` 一句丢弃 ⇒ 改写静默失效却照常发请求。
///
/// 上游传输层只承载 JSON（`upstream::send` 走 `req.json`），所以：`Text` 按其是否为合法
/// JSON 解析后采用，`Binary` 与解析失败的 `Text` 一律显式报错（不再假装成功），`Empty`
/// 保持 Adapter 原 body（要发 null 请写回 `Json(Value::Null)`）。
fn apply_outbound(up: &mut UpstreamRequest, outbound: RawMessage) -> Result<()> {
    if let Some(m) = &outbound.method {
        up.method = http::Method::from_bytes(m.as_bytes())
            .map_err(|_| GatewayError::Other(format!("插件写回非法 HTTP method: {m}")))?;
    }
    if let Some(url) = outbound.url {
        up.url = url;
    }
    up.headers = outbound.headers;
    match outbound.body {
        RawBody::Json { value } => up.body = value,
        RawBody::Text { text } => {
            up.body = serde_json::from_str(&text).map_err(|e| {
                GatewayError::Other(format!(
                    "插件写回的 body 文本不是合法 JSON，无法经 JSON 通道发往上游: {e}"
                ))
            })?;
        }
        RawBody::Binary { .. } => {
            return Err(GatewayError::Other(
                "上游请求只支持 JSON body，插件不得写回二进制".to_string(),
            ));
        }
        RawBody::Empty => {}
    }
    Ok(())
}

/// 路由前后的字段差异用参数表达：路由前上游信息尚不存在 ⇒ 传空串 / `None` / `Null`。
#[allow(clippy::too_many_arguments)]
fn new_trace(
    ctx: &ReqCtx,
    client_request: Value,
    stream: bool,
    upstream_model: String,
    provider_key: String,
    upstream_protocol: Option<Protocol>,
    upstream_request: Value,
) -> TraceRecord {
    TraceRecord {
        request_id: ctx.request_id.clone(),
        created_at: trace::now_ms(),
        session_id: ctx.session_id.clone(),
        model_alias: ctx.model_alias.clone(),
        upstream_model,
        provider_key,
        client_protocol: ctx.client_protocol.to_string(),
        upstream_protocol: upstream_protocol.map(|p| p.to_string()).unwrap_or_default(),
        stream,
        status: "ok".to_string(),
        latency_ms: 0,
        ttft_ms: None,
        // 插件重试次数由调用方在拿到 plugin_round 后填入（此处是路由前的骨架，恒为 0）
        retries: 0,
        usage: TraceUsage::default(),
        client_request,
        upstream_request,
        upstream_response: Value::Null,
        client_response: Value::Null,
        error: None,
    }
}

/// 路由之前（入站请求钩子阶段）的裸 trace：只有客户端请求侧信息。
///
/// `stream` 从（可能已被入站钩子改写的）请求体里读，读不到按非流式记。
fn pre_route_trace(ctx: &ReqCtx, inbound: &RawMessage) -> TraceRecord {
    let json = inbound.body.as_json().cloned().unwrap_or(Value::Null);
    let stream = json
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    new_trace(
        ctx,
        json,
        stream,
        String::new(),
        String::new(),
        None,
        Value::Null,
    )
}

/// 出站请求阶段（尚未发送）的 trace：路由已定，上游侧只有出站报文。
fn outbound_trace(
    ctx: &ReqCtx,
    outbound: &RawMessage,
    resolved: &ResolvedRoute,
    client_request: &Value,
) -> TraceRecord {
    new_trace(
        ctx,
        client_request.clone(),
        ctx.stream,
        resolved.upstream_model.clone(),
        resolved.provider_key.clone(),
        Some(outbound.protocol),
        serde_json::json!({
            "method": outbound.method,
            "url": outbound.url.as_deref().map(redact_url),
            "headers": redact_headers(&outbound.headers),
            "body": body_snapshot(&outbound.body),
        }),
    )
}

/// 已确定路由但尚未发出上游请求时的 trace（钩子/转换失败路径用）。
fn routed_trace(ctx: &ReqCtx, resolved: &ResolvedRoute, client_request: &Value) -> TraceRecord {
    new_trace(
        ctx,
        client_request.clone(),
        ctx.stream,
        resolved.upstream_model.clone(),
        resolved.provider_key.clone(),
        Some(resolved.protocol),
        Value::Null,
    )
}

/// 入口报文已取、路由尚未开始阶段的 trace（to_core_request/路由失败路径用）。
fn pre_core_trace(ctx: &ReqCtx, client_request: &Value) -> TraceRecord {
    let stream = client_request
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    new_trace(
        ctx,
        client_request.clone(),
        stream,
        String::new(),
        String::new(),
        None,
        Value::Null,
    )
}

/// 上游请求的 trace 快照（URL/headers 脱敏，body 原样——由 strip_bodies 统一抹除）。
fn upstream_request_snapshot(u: &UpstreamRequest) -> Value {
    serde_json::json!({
        "method": u.method.to_string(),
        "url": redact_url(&u.url),
        "headers": redact_headers(&u.headers),
        "body": u.body.clone(),
    })
}

/// 读取上游响应体并施加大小上限：非流式回程原先 `resp.bytes()` 无界读取，
/// 异常/恶意上游可让进程内存膨胀。超限与读取失败都以 502 返回。
async fn read_body_capped(resp: reqwest::Response, max: usize) -> Result<Vec<u8>> {
    use futures::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| GatewayError::Upstream {
            status: 502,
            message: format!("读取上游响应体失败: {e}"),
        })?;
        if buf.len() + chunk.len() > max {
            return Err(GatewayError::Upstream {
                status: 502,
                message: format!("上游响应体超过 {max} 字节上限"),
            });
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// 截断字符串到至多 `max` 字节（错误消息中的上游响应摘录用）。
fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

/// `RawBody` → trace 快照。非 JSON 也尽力留痕（不追求可反解析）。
fn body_snapshot(body: &RawBody) -> Value {
    match body {
        RawBody::Json { value } => value.clone(),
        RawBody::Text { text } => Value::String(text.clone()),
        RawBody::Binary { data } => serde_json::json!({ "binary_bytes": data.len() }),
        RawBody::Empty => Value::Null,
    }
}

/// trace 快照中的请求头脱敏：疑似鉴权头的值整体替换为 `[REDACTED]`，
/// 防止 Bearer Token / API Key / Cookie 泄漏进写入磁盘的 trace 文件。
fn redact_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(name, value)| {
            let n = name.to_ascii_lowercase();
            let sensitive = n.contains("auth")
                || n.contains("api-key")
                || n.contains("apikey")
                || n.contains("token")
                || n.contains("cookie")
                || n.contains("secret");
            if sensitive {
                (name.clone(), "[REDACTED]".to_string())
            } else {
                (name.clone(), value.clone())
            }
        })
        .collect()
}

/// URL 查询参数中的密钥脱敏（如 Google 风格 `?key=API_KEY`）。
fn redact_url(url: &str) -> String {
    let Some((base, query)) = url.split_once('?') else {
        return url.to_string();
    };
    let redacted: Vec<String> = query
        .split('&')
        .map(|pair| {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            let kl = k.to_ascii_lowercase();
            if kl.contains("key") || kl.contains("token") || kl.contains("secret") {
                format!("{k}=[REDACTED]")
            } else if v.is_empty() {
                k.to_string()
            } else {
                format!("{k}={v}")
            }
        })
        .collect();
    format!("{base}?{}", redacted.join("&"))
}

/// 收尾落审计：一行 usage + 一份 trace 文件。
///
/// `trace` 的 `status` 与各报文快照由调用方**事先填好**，错误消息取 `trace.error`；
/// `usage_status` 单独传是因为二者的取值口径不同——例如插件以 200 短路直答时，
/// trace 记 `short_circuit`（说明是谁答的），usage 记 `ok`（对客户端而言确实成功）。
fn finish_audit(
    state: &Arc<AppState>,
    ctx: &ReqCtx,
    trace: &mut TraceRecord,
    start: Instant,
    usage: &Usage,
    usage_status: &str,
) {
    // 请求/响应体可选记录：关闭时只留元数据（方法/URL/头/用量），体一律不写入磁盘。
    // 在统一收尾处抹除，覆盖成功/错误/短路/故障转移全部路径。
    if !state.config.trace_record_bodies {
        crate::trace::strip_bodies(trace);
    }
    usage::record(
        &state.db,
        ctx,
        &trace.upstream_model,
        usage,
        usage_status,
        trace.error.as_deref(),
        start,
        None,
    );
    trace.latency_ms = start.elapsed().as_millis() as u64;
    trace::write(
        state.config.trace_dir.as_deref(),
        trace,
        state.config.trace_retention,
    );
}

/// 插件短路直答：**照常**落 usage 与 trace 后，把插件给的报文回给客户端。
///
/// 历史缺陷是这些路径直接 `return`，插件代答的请求在用量统计与 Traces 页完全不可见。
#[allow(clippy::too_many_arguments)]
fn answered(
    state: &Arc<AppState>,
    ctx: &ReqCtx,
    start: Instant,
    mut trace: TraceRecord,
    usage: Usage,
    status: u16,
    headers: Vec<(String, String)>,
    body: RawBody,
) -> Response {
    trace.status = "short_circuit".to_string();
    trace.error = Some(format!("插件短路直答（HTTP {status}）"));
    trace.client_response = body_snapshot(&body);
    let usage_status = if status < 400 { "ok" } else { "error" };
    finish_audit(state, ctx, &mut trace, start, &usage, usage_status);
    short_circuit(status, headers, body)
}

/// 请求处理中途失败的统一收尾：落 usage + trace 后返回原错误。
/// 让各 `?` 点位写成 `.map_err(|e| fail_audit(...))?`，不留无审计的失败路径。
fn fail_audit(
    state: &Arc<AppState>,
    ctx: &ReqCtx,
    start: Instant,
    mut trace: TraceRecord,
    e: GatewayError,
) -> GatewayError {
    trace.status = "error".to_string();
    trace.error = Some(e.to_string());
    finish_audit(state, ctx, &mut trace, start, &Usage::default(), "error");
    e
}

/// 插件主动中止：同样留下审计痕迹，再把错误交回上层。返回错误而非 `()`，
/// 让调用点写成 `return Err(aborted(..))`，不给人「忘了落审计」留空间。
fn aborted(
    state: &Arc<AppState>,
    ctx: &ReqCtx,
    start: Instant,
    mut trace: TraceRecord,
    message: String,
) -> GatewayError {
    trace.status = "aborted".to_string();
    trace.error = Some(message.clone());
    finish_audit(state, ctx, &mut trace, start, &Usage::default(), "error");
    GatewayError::Other(message)
}

/// header 名/值来自插件，可能是非法字符序列——`Response::builder` 对非法
/// header 静默吞错但会把 builder 置为失败态，随后 `body()` 返回 Err，
/// 原先在此 `.unwrap()` 直接 panic。非法头丢弃 + 构造失败回退 500。
fn short_circuit(status: u16, headers: Vec<(String, String)>, body: RawBody) -> Response {
    let code = StatusCode::from_u16(status).unwrap_or(StatusCode::OK);
    let body_bytes = match body {
        RawBody::Json { value } => value.to_string().into_bytes(),
        RawBody::Text { text } => text.into_bytes(),
        RawBody::Binary { data } => data,
        RawBody::Empty => Vec::new(),
    };
    let has_ct = headers
        .iter()
        .any(|(k, _)| k.eq_ignore_ascii_case("content-type"));
    let mut builder = Response::builder().status(code);
    if !has_ct {
        builder = builder.header("content-type", "application/json");
    }
    for (k, v) in headers {
        match (
            axum::http::HeaderName::from_bytes(k.as_bytes()),
            axum::http::HeaderValue::from_str(&v),
        ) {
            (Ok(name), Ok(value)) => {
                builder = builder.header(name, value);
            }
            _ => tracing::warn!(header = %k, "插件写回非法响应头，已丢弃"),
        }
    }
    builder.body(Body::from(body_bytes)).unwrap_or_else(|_| {
        Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from(
                r#"{"error":{"message":"插件短路响应构造失败"}}"#,
            ))
            .unwrap_or_else(|_| Response::new(Body::empty()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn redacts_sensitive_headers_and_url_keys() {
        let headers = vec![
            ("content-type".to_string(), "application/json".to_string()),
            (
                "Authorization".to_string(),
                "Bearer sk-ant-secret".to_string(),
            ),
            ("x-api-key".to_string(), "sk-123".to_string()),
            ("X-Goog-Api-Key".to_string(), "goog-key".to_string()),
            ("Cookie".to_string(), "session=abc".to_string()),
        ];
        let redacted = redact_headers(&headers);
        assert_eq!(redacted[0].1, "application/json"); // 非鉴权头原样保留
        for (name, value) in &redacted[1..] {
            assert_eq!(value, "[REDACTED]", "{name} 应被脱敏");
        }
    }

    #[test]
    fn redacts_key_query_params_but_keeps_others() {
        assert_eq!(
            redact_url("https://x/v1?a=1&key=API_KEY&b=2"),
            "https://x/v1?a=1&key=[REDACTED]&b=2"
        );
        assert_eq!(redact_url("https://x/v1/messages"), "https://x/v1/messages");
    }

    fn upstream_req() -> UpstreamRequest {
        UpstreamRequest {
            method: http::Method::POST,
            url: "https://up.example/v1/messages".into(),
            headers: vec![("content-type".into(), "application/json".into())],
            body: json!({ "model": "original" }),
            stream: false,
        }
    }

    /// 由 Adapter 产出的请求构造出站报文（与 dispatch 里的组装方式一致）。
    fn outbound_from(up: &UpstreamRequest) -> RawMessage {
        RawMessage {
            stage: RawStage::UpstreamRequest,
            protocol: Protocol::Anthropic,
            provider: Some("p".into()),
            method: Some(up.method.to_string()),
            url: Some(up.url.clone()),
            status: None,
            headers: up.headers.clone(),
            body: RawBody::json(up.body.clone()),
            error: false,
            session_id: None,
        }
    }

    /// 回归：插件对 method 的改写必须生效。旧代码把 method 写进 Lua 表却从不回读，
    /// 改 `msg.method` 静默无效。
    #[test]
    fn reads_back_method() {
        let mut u = upstream_req();
        let mut ob = outbound_from(&u);
        ob.method = Some("PUT".into());
        assert!(apply_outbound(&mut u, ob).is_ok());
        assert_eq!(u.method, http::Method::PUT);
    }

    #[test]
    fn url_and_headers_and_json_body_are_read_back() {
        let mut u = upstream_req();
        let mut ob = outbound_from(&u);
        ob.url = Some("https://other.example/rewritten".into());
        ob.headers = vec![("x-tapped".into(), "1".into())];
        ob.body = RawBody::json(json!({ "model": "rewritten" }));
        assert!(apply_outbound(&mut u, ob).is_ok());
        assert_eq!(u.url, "https://other.example/rewritten");
        assert_eq!(u.headers, vec![("x-tapped".into(), "1".into())]);
        assert_eq!(u.body["model"], "rewritten");
    }

    /// 非法 method 必须报错，而不是悄悄按原方法发出。
    #[test]
    fn invalid_method_is_rejected() {
        let mut u = upstream_req();
        let mut ob = outbound_from(&u);
        ob.method = Some("BAD METHOD".into());
        assert!(apply_outbound(&mut u, ob).is_err());
    }

    /// 插件把 body 重写成文本：是合法 JSON 就采用——旧代码直接丢弃。
    #[test]
    fn text_body_is_accepted_when_it_is_json() {
        let mut u = upstream_req();
        let mut ob = outbound_from(&u);
        ob.body = RawBody::Text {
            text: r#"{"model":"from-text"}"#.into(),
        };
        assert!(apply_outbound(&mut u, ob).is_ok());
        assert_eq!(u.body["model"], "from-text");
    }

    /// 上游只走 JSON 通道：非 JSON 文本与二进制必须显式失败，不能假装改写成功。
    #[test]
    fn non_json_and_binary_body_are_rejected() {
        let mut u = upstream_req();
        let mut ob = outbound_from(&u);
        ob.body = RawBody::Text {
            text: "not json at all".into(),
        };
        assert!(apply_outbound(&mut u, ob).is_err(), "非 JSON 文本应报错");
        assert_eq!(u.body["model"], "original", "报错时不得留下半成品改写");

        let mut ob2 = outbound_from(&u);
        ob2.body = RawBody::Binary {
            data: vec![1, 2, 3],
        };
        assert!(apply_outbound(&mut u, ob2).is_err(), "二进制应报错");
    }

    /// `Empty` 表示插件没写 body，保持 Adapter 原 body（而非发 null）。
    #[test]
    fn empty_body_keeps_adapter_body() {
        let mut u = upstream_req();
        let mut ob = outbound_from(&u);
        ob.body = RawBody::Empty;
        assert!(apply_outbound(&mut u, ob).is_ok());
        assert_eq!(u.body["model"], "original");
    }

    /// 回归：`handle_request` 的 future 在返回响应前被丢（客户端断开/进程退出）
    /// 时，usage 与 trace 必须留下 aborted 记录——否则上游观测到的 client_gone
    /// 在本地无任何痕迹可查（历史上这类请求完全隐形）。
    #[test]
    fn inflight_audit_writes_aborted_on_drop() {
        let db = Arc::new(Database::open_in_memory().unwrap());
        let dir = std::env::temp_dir().join(format!("mb-inflight-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        {
            let _guard = InflightAudit::new(
                &db,
                Some(&dir.to_string_lossy()),
                true,
                0,
                Protocol::OpenAiChat,
                &json!({"model": "m1", "stream": true}),
            );
            // 不 disarm 直接析构，模拟 future 被取消
        }
        let recs = db
            .query_usage(&moonbridge_store::models::UsageQuery {
                status: Some("aborted".into()),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(recs.len(), 1, "被丢的 in-flight 请求应补记一条 aborted");
        assert_eq!(recs[0].model.as_deref(), Some("m1"));
        assert!(recs[0].error.as_deref().unwrap_or("").contains("被取消"));
        // trace 也应落盘（request_id 目录下）
        assert!(
            std::fs::read_dir(&dir)
                .map(|mut d| d.next().is_some())
                .unwrap_or(false),
            "trace 目录应有记录文件"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// disarm 后析构（正常返回路径）不得写任何记录——否则所有请求都会双写。
    #[test]
    fn inflight_audit_disarmed_writes_nothing() {
        let db = Arc::new(Database::open_in_memory().unwrap());
        InflightAudit::new(
            &db,
            None,
            true,
            0,
            Protocol::OpenAiChat,
            &json!({"model": "m1"}),
        )
        .disarm();
        assert_eq!(db.usage_summary().unwrap().requests, 0);
    }
}
