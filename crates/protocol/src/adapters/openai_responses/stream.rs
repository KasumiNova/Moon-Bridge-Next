//! OpenAI Responses 入口 Adapter —— 流式编码。
//!
//! 把协议中立的 [`CoreStreamEvent`] 编码为 Responses SSE 事件序列。采用无状态
//! 映射：`item_id` 由块 index 稳定派生，文本增量经 `response.output_text.delta`
//! 转发，结束时发 `response.completed`（携带 usage）。Codex CLI 依据 delta 累积
//! 文本，因此 `*.done` 事件的 text 字段可为空。

use async_trait::async_trait;
use moonbridge_core::{ContentBlock, CoreStreamEvent, Protocol, Result, StopReason, StreamDelta};
use serde_json::{json, Value};

use super::client::usage_from_value;
use super::dto::{self, event};
use super::OpenAiResponsesAdapter;
use crate::adapter::{
    ClientStreamAdapter, ProviderStreamAdapter, StreamDecodeState, StreamEncodeState,
};
use crate::context::ReqCtx;
use crate::raw::{ChunkStage, RawChunk};

fn sse(event_name: &str, data: Value) -> RawChunk {
    RawChunk::json(
        ChunkStage::ClientChunk,
        Protocol::OpenAiResponse,
        Some(event_name.to_string()),
        data,
    )
}

fn item_id(index: usize) -> String {
    format!("msg_{index}")
}

/// function_call item 的 `id`/`item_id`：优先用上游透传的真实 item id，
/// 上游未给（或非 Responses 上游）时由 output_index 稳定派生，
/// added / arguments.delta / arguments.done / done / completed 共用。
fn tool_item_id(st: &StreamEncodeState, index: usize) -> String {
    if let Some(ContentBlock::ToolUse {
        item_id: Some(id), ..
    }) = st.blocks.get(&index)
    {
        return id.clone();
    }
    format!("fc_{index}")
}

#[async_trait]
impl ClientStreamAdapter for OpenAiResponsesAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::OpenAiResponse
    }

    fn encode(
        &self,
        ctx: &ReqCtx,
        ev: &CoreStreamEvent,
        st: &mut StreamEncodeState,
    ) -> Result<Vec<RawChunk>> {
        let mut out = Vec::new();
        match ev {
            CoreStreamEvent::MessageStart { id, model } => {
                st.message_id = id.clone();
                out.push(sse(
                    event::CREATED,
                    json!({
                        "type": event::CREATED,
                        "response": dto::response_skeleton(id, model, "in_progress"),
                    }),
                ));
                out.push(sse(
                    event::IN_PROGRESS,
                    json!({
                        "type": event::IN_PROGRESS,
                        "response": dto::response_skeleton(id, model, "in_progress"),
                    }),
                ));
            }
            CoreStreamEvent::BlockStart { index, block } => {
                st.note_start(*index, block);
                match block {
                    ContentBlock::ToolUse { id, name, .. } => {
                        out.push(sse(
                            event::OUTPUT_ITEM_ADDED,
                            json!({
                                "type": event::OUTPUT_ITEM_ADDED,
                                "output_index": index,
                                "item": dto::function_call_item(
                                    &tool_item_id(st, *index), id, name, "", "in_progress"
                                ),
                            }),
                        ));
                    }
                    ContentBlock::Reasoning { .. } => {
                        // reasoning item：收尾（output_item.done）在 BlockStop 按
                        // block 类型分派，这里只发 item 起始
                        out.push(sse(
                        event::OUTPUT_ITEM_ADDED,
                        json!({
                            "type": event::OUTPUT_ITEM_ADDED,
                            "output_index": index,
                            "item": { "type": "reasoning", "id": item_id(*index), "summary": [] },
                        }),
                    ));
                    }
                    _ => {
                        out.push(sse(
                            event::OUTPUT_ITEM_ADDED,
                            json!({
                                "type": event::OUTPUT_ITEM_ADDED,
                                "output_index": index,
                                "item": dto::message_item(&item_id(*index), "", "in_progress"),
                            }),
                        ));
                        out.push(sse(
                            event::CONTENT_PART_ADDED,
                            json!({
                                "type": event::CONTENT_PART_ADDED,
                                "item_id": item_id(*index),
                                "output_index": index,
                                "content_index": 0,
                                "part": { "type": "output_text", "text": "", "annotations": [] },
                            }),
                        ));
                    }
                }
            }
            CoreStreamEvent::BlockDelta { index, delta } => match delta {
                StreamDelta::Text { text } => {
                    st.push_delta(*index, text);
                    out.push(sse(
                        event::TEXT_DELTA,
                        json!({
                            "type": event::TEXT_DELTA,
                            "item_id": item_id(*index),
                            "output_index": index,
                            "content_index": 0,
                            "delta": text,
                        }),
                    ));
                }
                StreamDelta::ToolInput { partial_json } => {
                    st.push_delta(*index, partial_json);
                    out.push(sse(
                        event::FUNC_ARGS_DELTA,
                        json!({
                            "type": event::FUNC_ARGS_DELTA,
                            "item_id": tool_item_id(st, *index),
                            "output_index": index,
                            "delta": partial_json,
                        }),
                    ));
                }
                StreamDelta::Reasoning { text } => {
                    st.push_delta(*index, text);
                    // 摘要明文增量：客户端累积后随 output_item.done 的 item 一致。
                    // summary_index 必填——严格校验的客户端缺它丢弃整个增量。
                    out.push(sse(
                        event::REASONING_SUMMARY_TEXT_DELTA,
                        json!({
                            "type": event::REASONING_SUMMARY_TEXT_DELTA,
                            "item_id": item_id(*index),
                            "output_index": index,
                            "summary_index": 0,
                            "delta": text,
                        }),
                    ));
                }
                StreamDelta::ReasoningSignature { signature } => {
                    // 凭据不经增量事件下发：记入起始块，随 BlockStop 的
                    // reasoning item（output_item.done）原样携带
                    if let Some(ContentBlock::Reasoning { signature: s, .. }) =
                        st.blocks.get_mut(index)
                    {
                        *s = Some(signature.clone());
                    }
                }
            },
            CoreStreamEvent::BlockStop { index, block } => {
                // 收尾块优先取事件携带值，回退到 BlockStart 记录的起始块——
                // Anthropic 等上游的 BlockStop 不带 block，但工具调用的收尾
                // 事件必须还原出 function_call 形态（否则客户端拿不到完整
                // arguments，工具调用静默丢失）。
                let resolved = block.clone().or_else(|| st.blocks.get(index).cloned());
                match resolved {
                    Some(ContentBlock::Reasoning {
                        text, signature, ..
                    }) => {
                        let mut item = json!({
                            "type": "reasoning",
                            "id": item_id(*index),
                            "summary": [{ "type": "summary_text", "text": text }],
                        });
                        // 本家凭据还原原文；异源凭据带标记原样下发（opaque，
                        // 客户端存入历史、回传后回到归属协议再解除标记）。
                        if let Some(enc) = signature.as_deref().map(|s| {
                            crate::adapters::emit_signature(crate::adapters::SIG_OPENAI, s)
                        }) {
                            if !enc.is_empty() {
                                item["encrypted_content"] = json!(enc);
                            }
                        }
                        out.push(sse(
                            event::OUTPUT_ITEM_DONE,
                            json!({
                                "type": event::OUTPUT_ITEM_DONE,
                                "output_index": index,
                                "item": item,
                            }),
                        ));
                    }
                    Some(ContentBlock::ToolUse {
                        id, name, input, ..
                    }) => {
                        // function_call 收尾序列：arguments.done（完整参数串）+
                        // output_item.done（function_call item）。缺了它们，
                        // 客户端（Codex）永远不会派发这个工具调用。
                        let args = st
                            .delta_acc
                            .get(index)
                            .cloned()
                            .filter(|s| !s.is_empty())
                            .unwrap_or_else(|| {
                                if input.is_object()
                                    && input.as_object().map(|o| o.is_empty()).unwrap_or(true)
                                {
                                    "{}".to_string()
                                } else {
                                    input.to_string()
                                }
                            });
                        out.push(sse(
                            event::FUNC_ARGS_DONE,
                            json!({
                                "type": event::FUNC_ARGS_DONE,
                                "item_id": tool_item_id(st, *index),
                                "output_index": index,
                                "name": name,
                                "arguments": args,
                            }),
                        ));
                        out.push(sse(
                            event::OUTPUT_ITEM_DONE,
                            json!({
                                "type": event::OUTPUT_ITEM_DONE,
                                "output_index": index,
                                "item": dto::function_call_item(
                                    &tool_item_id(st, *index), &id, &name, &args, "completed"
                                ),
                            }),
                        ));
                    }
                    _ => {
                        let text = st.delta_acc.get(index).cloned().unwrap_or_default();
                        out.push(sse(
                            event::TEXT_DONE,
                            json!({
                                "type": event::TEXT_DONE,
                                "item_id": item_id(*index),
                                "output_index": index,
                                "content_index": 0,
                                "text": text,
                            }),
                        ));
                        out.push(sse(
                            event::CONTENT_PART_DONE,
                            json!({
                                "type": event::CONTENT_PART_DONE,
                                "item_id": item_id(*index),
                                "output_index": index,
                                "content_index": 0,
                                "part": { "type": "output_text", "text": text, "annotations": [] },
                            }),
                        ));
                        out.push(sse(
                            event::OUTPUT_ITEM_DONE,
                            json!({
                                "type": event::OUTPUT_ITEM_DONE,
                                "output_index": index,
                                "item": dto::message_item(&item_id(*index), &text, "completed"),
                            }),
                        ));
                    }
                }
            }
            CoreStreamEvent::MessageDelta { stop_reason, usage } => {
                if let Some(u) = usage {
                    st.merge_usage(u);
                }
                // 无 stop_reason 的 MessageDelta 是中途用量上报（如 Anthropic
                // message_start 携带的 input），不是终态——此时绝不能发
                // response.completed，否则客户端在流中段就以为回合已结束。
                let Some(reason) = stop_reason else {
                    return Ok(out);
                };
                // 用量取跨 delta 合并后的累计值：Anthropic 上游把 input 放在
                // message_start、output 放在收尾 message_delta，单看末个 delta
                // 只有半份。
                // usage 是终态 response 的必填对象：上游未上报时也要发
                // 零值对象，发 null 会让严格校验的客户端丢掉整个
                // response.completed（finish 事件随之丢失）。
                let usage = dto::usage_object(&st.usage_acc);
                // 截断终止 → response.incomplete（缺省视为 completed）。
                let (ty, status, detail) = match reason {
                    StopReason::MaxTokens => {
                        (event::INCOMPLETE, "incomplete", Some("max_output_tokens"))
                    }
                    StopReason::ContentFilter => {
                        (event::INCOMPLETE, "incomplete", Some("content_filter"))
                    }
                    _ => (event::COMPLETED, "completed", None),
                };
                let mut resp = dto::response_skeleton(&st.message_id, &ctx.model_alias, status);
                if let Some(obj) = resp.as_object_mut() {
                    obj.insert("usage".to_string(), usage);
                    // 终态 response 携带组装好的 output——只读 response.completed
                    // 而不逐 delta 累积的客户端，拿它当最终结果。
                    obj.insert("output".to_string(), Value::Array(assembled_output(st)));
                    if let Some(d) = detail {
                        obj.insert("incomplete_details".to_string(), json!({ "reason": d }));
                    }
                }
                out.push(sse(ty, json!({ "type": ty, "response": resp })));
            }
            CoreStreamEvent::MessageStop => {}
            CoreStreamEvent::Error { message } => {
                out.push(sse(
                    "response.failed",
                    json!({
                        "type": "response.failed",
                        "response": { "status": "failed", "error": { "message": message } },
                    }),
                ));
            }
        }
        Ok(out)
    }
}

/// 从 chunk 取出 JSON data（优先已解析的 data，回退解析 raw 文本）。
fn chunk_json(chunk: &RawChunk) -> Option<Value> {
    if let Some(v) = chunk.data.as_json() {
        return Some(v.clone());
    }
    serde_json::from_str::<Value>(&chunk.raw).ok()
}

fn output_index(data: &Value) -> usize {
    data.get("output_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize
}

/// 由 encode 状态组装终态 `response.output`：按 output_index 顺序，
/// 用 BlockStart 记下的块类型 + delta_acc 累积的完整载荷还原各 item。
/// 只读 `response.completed` 的客户端（而非逐 delta 累积）依赖它拿最终结果。
fn assembled_output(st: &StreamEncodeState) -> Vec<Value> {
    let mut items: Vec<(usize, Value)> = Vec::new();
    for (&index, block) in &st.blocks {
        let acc = st.delta_acc.get(&index).cloned().unwrap_or_default();
        let item = match block {
            ContentBlock::Reasoning {
                text, signature, ..
            } => {
                let text = if acc.is_empty() { text.clone() } else { acc };
                let mut item = json!({
                    "type": "reasoning",
                    "id": item_id(index),
                    "summary": [{ "type": "summary_text", "text": text }],
                });
                if let Some(enc) = signature
                    .as_deref()
                    .map(|s| crate::adapters::emit_signature(crate::adapters::SIG_OPENAI, s))
                {
                    if !enc.is_empty() {
                        item["encrypted_content"] = json!(enc);
                    }
                }
                item
            }
            ContentBlock::ToolUse {
                id, name, input, ..
            } => {
                let args = if acc.is_empty() {
                    if input.is_object() && input.as_object().map(|o| o.is_empty()).unwrap_or(true)
                    {
                        "{}".to_string()
                    } else {
                        input.to_string()
                    }
                } else {
                    acc
                };
                dto::function_call_item(&tool_item_id(st, index), id, name, &args, "completed")
            }
            ContentBlock::Text { text } => {
                let text = if acc.is_empty() { text.clone() } else { acc };
                dto::message_item(&item_id(index), &text, "completed")
            }
            _ => continue,
        };
        items.push((index, item));
    }
    items.sort_by_key(|(i, _)| *i);
    items.into_iter().map(|(_, v)| v).collect()
}

#[async_trait]
impl ProviderStreamAdapter for OpenAiResponsesAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::OpenAiResponse
    }

    /// Responses SSE → Core 流事件（与入口 encode 互为逆向）。
    fn decode(
        &self,
        _ctx: &ReqCtx,
        _st: &mut StreamDecodeState,
        chunk: &RawChunk,
    ) -> Result<Vec<CoreStreamEvent>> {
        let Some(data) = chunk_json(chunk) else {
            return Ok(Vec::new());
        };
        let ty = data.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let mut out = Vec::new();

        match ty {
            "response.created" => {
                let resp = data.get("response").cloned().unwrap_or(Value::Null);
                let id = resp
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let model = resp
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                out.push(CoreStreamEvent::MessageStart { id, model });
            }
            "response.output_item.added" => {
                let item = data.get("item").cloned().unwrap_or(Value::Null);
                let block = match item.get("type").and_then(|t| t.as_str()) {
                    Some("function_call") => ContentBlock::ToolUse {
                        id: item
                            .get("call_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        item_id: item.get("id").and_then(|v| v.as_str()).map(String::from),
                        name: item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        namespace: None,
                        input: json!({}),
                        signature: None,
                    },
                    // reasoning item：后续 summary/reasoning 文本增量挂载到同一 output_index
                    Some("reasoning") => ContentBlock::Reasoning {
                        text: String::new(),
                        signature: None,
                        redacted: false,
                    },
                    _ => ContentBlock::text(""),
                };
                out.push(CoreStreamEvent::BlockStart {
                    index: output_index(&data),
                    block,
                });
            }
            // 推理增量：官方摘要（summary_text）与原文（reasoning_text，第三方兼容实现）
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let text = data
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                if !text.is_empty() {
                    out.push(CoreStreamEvent::BlockDelta {
                        index: output_index(&data),
                        delta: StreamDelta::Reasoning { text },
                    });
                }
            }
            "response.output_text.delta" => {
                let text = data
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                out.push(CoreStreamEvent::BlockDelta {
                    index: output_index(&data),
                    delta: StreamDelta::Text { text },
                });
            }
            "response.function_call_arguments.delta" => {
                let partial = data
                    .get("delta")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                out.push(CoreStreamEvent::BlockDelta {
                    index: output_index(&data),
                    delta: StreamDelta::ToolInput {
                        partial_json: partial,
                    },
                });
            }
            "response.output_item.done" => {
                let item = data.get("item").cloned().unwrap_or(Value::Null);
                let index = output_index(&data);
                let mut block = None;
                match item.get("type").and_then(|t| t.as_str()) {
                    Some("reasoning") => {
                        // reasoning item 的明文/凭据在 done 事件的 item 上：组装进收尾块，
                        // encode 端据此发 reasoning 形态的收尾事件（凭据原样带回客户端）。
                        let mut text = String::new();
                        for key in ["summary", "content"] {
                            if let Some(arr) = item.get(key).and_then(|s| s.as_array()) {
                                for x in arr {
                                    if let Some(t) = x.get("text").and_then(|t| t.as_str()) {
                                        text.push_str(t);
                                    }
                                }
                            }
                        }
                        let signature = crate::adapters::tag_signature(
                            crate::adapters::SIG_OPENAI,
                            item.get("encrypted_content")
                                .and_then(|v| v.as_str())
                                .map(String::from),
                        );
                        // 无明文且无凭据：不发 reasoning 形态收尾（encode 回退 message 形态）
                        if !text.is_empty() || signature.is_some() {
                            block = Some(ContentBlock::Reasoning {
                                text,
                                signature,
                                redacted: false,
                            });
                        }
                    }
                    // function_call item：done 携带完整 arguments，必须还原成
                    // ToolUse 块随 BlockStop 下发——下游 encode 需要它产出
                    // function_call 形态的收尾事件。
                    Some("function_call") => {
                        let input = item
                            .get("arguments")
                            .and_then(|a| a.as_str())
                            .and_then(|s| serde_json::from_str::<Value>(s).ok())
                            .unwrap_or_else(|| json!({}));
                        block = Some(ContentBlock::ToolUse {
                            id: item
                                .get("call_id")
                                .or_else(|| item.get("id"))
                                .and_then(|v| v.as_str())
                                .unwrap_or_default()
                                .to_string(),
                            item_id: item.get("id").and_then(|v| v.as_str()).map(String::from),
                            name: item
                                .get("name")
                                .and_then(|v| v.as_str())
                                .unwrap_or_default()
                                .to_string(),
                            namespace: None,
                            input,
                            signature: None,
                        });
                    }
                    Some("message") => {
                        let mut text = String::new();
                        if let Some(parts) = item.get("content").and_then(|c| c.as_array()) {
                            for p in parts {
                                if let Some(t) = p.get("text").and_then(|t| t.as_str()) {
                                    text.push_str(t);
                                }
                            }
                        }
                        block = Some(ContentBlock::text(text));
                    }
                    _ => {}
                }
                // 凭据以增量先行：anthropic/chat 入口仅消费凭据增量（signature_delta /
                // <mb-cot> 追加），必须在收尾前到达；responses 入口忽略此增量，
                // 凭据随下方 BlockStop 的 reasoning item 发出。
                if let Some(ContentBlock::Reasoning {
                    signature: Some(sig),
                    ..
                }) = &block
                {
                    out.push(CoreStreamEvent::BlockDelta {
                        index,
                        delta: StreamDelta::ReasoningSignature {
                            signature: sig.clone(),
                        },
                    });
                }
                out.push(CoreStreamEvent::BlockStop { index, block });
            }
            "response.completed" | "response.incomplete" => {
                let resp = data.get("response").cloned().unwrap_or(Value::Null);
                let usage = resp.get("usage").map(usage_from_value);
                // incomplete 的早停原因落在 incomplete_details.reason；
                // 没有细节时按正常结束处理。
                let reason = if ty == "response.incomplete" {
                    match resp
                        .get("incomplete_details")
                        .and_then(|d| d.get("reason"))
                        .and_then(|r| r.as_str())
                    {
                        Some("max_output_tokens") => Some(StopReason::MaxTokens),
                        Some("content_filter") => Some(StopReason::ContentFilter),
                        _ => Some(StopReason::EndTurn),
                    }
                } else {
                    Some(StopReason::EndTurn)
                };
                out.push(CoreStreamEvent::MessageDelta {
                    stop_reason: reason,
                    usage,
                });
                out.push(CoreStreamEvent::MessageStop);
            }
            "response.failed" => {
                let message = data
                    .get("response")
                    .and_then(|r| r.get("error"))
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("response failed")
                    .to_string();
                out.push(CoreStreamEvent::Error { message });
            }
            // 独立的 `error` 事件帧：new-api 类中转在 response.failed 之前先发它，
            // 携带真实失败原因（rate_limit_error/turn_rate_limit_exceeded + Retry-After），
            // 比 response.failed 的通用包装（"Upstream request failed"）信息量大。
            // 消息合成 `code: message`，code 留给插件 match 与日志分辨。
            "error" => {
                let code = data
                    .get("error")
                    .and_then(|e| e.get("code"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                let message = data
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();
                if !message.is_empty() {
                    let mut text = if code.is_empty() {
                        message.to_string()
                    } else {
                        format!("{code}: {message}")
                    };
                    // 限流类错误把上游建议的等待时长带出来（`[retry-after=Ns]`），
                    // 插件据此排定重试延迟——固定退避对容量窗口毫无意义。
                    if let Some(ra) = data
                        .get("headers")
                        .and_then(|h| h.get("Retry-After"))
                        .and_then(|v| v.as_str())
                        .filter(|s| !s.is_empty())
                    {
                        text = format!("{text} [retry-after={ra}s]");
                    }
                    out.push(CoreStreamEvent::Error { message: text });
                }
            }
            _ => {}
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(v: Value) -> RawChunk {
        RawChunk::json(ChunkStage::UpstreamChunk, Protocol::OpenAiResponse, None, v)
    }

    #[test]
    fn decodes_reasoning_item_and_deltas() {
        let adapter = OpenAiResponsesAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);

        // reasoning item 开始：块类型为 Reasoning（而非默认 text 块）
        let evs = adapter
            .decode(
                &ctx,
                &mut StreamDecodeState::default(),
                &chunk(json!({
                    "type": "response.output_item.added", "output_index": 0,
                    "item": { "type": "reasoning", "summary": [] }
                })),
            )
            .unwrap();
        assert!(matches!(
            &evs[0],
            CoreStreamEvent::BlockStart {
                block: ContentBlock::Reasoning { .. },
                ..
            }
        ));

        // 官方摘要与第三方原文两种增量都解析为 Reasoning
        for ty in [
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ] {
            let evs = adapter
                .decode(
                    &ctx,
                    &mut StreamDecodeState::default(),
                    &chunk(json!({ "type": ty, "output_index": 0, "delta": "hmm" })),
                )
                .unwrap();
            match &evs[0] {
                CoreStreamEvent::BlockDelta {
                    delta: StreamDelta::Reasoning { text },
                    ..
                } => {
                    assert_eq!(text, "hmm")
                }
                other => panic!("expected reasoning delta for {ty}, got {other:?}"),
            }
        }
    }

    #[test]
    fn decodes_responses_stream() {
        let adapter = OpenAiResponsesAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);

        let evs = adapter
            .decode(
                &ctx,
                &mut StreamDecodeState::default(),
                &chunk(
                    json!({"type":"response.created","response":{"id":"resp_1","model":"gpt-x"}}),
                ),
            )
            .unwrap();
        assert!(matches!(evs[0], CoreStreamEvent::MessageStart { .. }));

        let evs = adapter
            .decode(
                &ctx,
                &mut StreamDecodeState::default(),
                &chunk(json!({"type":"response.output_text.delta","output_index":0,"delta":"Hi"})),
            )
            .unwrap();
        match &evs[0] {
            CoreStreamEvent::BlockDelta {
                delta: StreamDelta::Text { text },
                ..
            } => {
                assert_eq!(text, "Hi")
            }
            _ => panic!("expected text delta"),
        }

        let evs = adapter
            .decode(&ctx, &mut StreamDecodeState::default(), &chunk(json!({"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3}}})))
            .unwrap();
        assert!(matches!(evs[0], CoreStreamEvent::MessageDelta { .. }));
        assert!(matches!(evs[1], CoreStreamEvent::MessageStop));
    }

    /// 加密 CoT 流式 round-trip：上游 output_item.done 携带 encrypted_content →
    /// BlockStop 的 reasoning 块凭据 → 入口 encode 以 reasoning item 形态原样下发。
    #[test]
    fn reasoning_encrypted_content_roundtrips_via_stop() {
        let adapter = OpenAiResponsesAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);

        // 上游：reasoning item 只带 encrypted_content（无任何明文摘要）
        let evs = adapter
            .decode(
                &ctx,
                &mut StreamDecodeState::default(),
                &chunk(json!({
                    "type": "response.output_item.done", "output_index": 0,
                    "item": { "type": "reasoning", "summary": [], "encrypted_content": "ENC" }
                })),
            )
            .unwrap();
        // 凭据以增量先行（anthropic/chat 入口消费），BlockStop 仍携带完整块
        match &evs[0] {
            CoreStreamEvent::BlockDelta {
                delta: StreamDelta::ReasoningSignature { signature },
                ..
            } => {
                assert_eq!(signature, "oai:ENC", "凭据带来源标记");
            }
            other => panic!("expected signature delta, got {other:?}"),
        }
        match &evs[1] {
            CoreStreamEvent::BlockStop {
                block:
                    Some(ContentBlock::Reasoning {
                        text,
                        signature: Some(enc),
                        ..
                    }),
                ..
            } => {
                assert!(text.is_empty(), "encrypted 不是展示文本");
                assert_eq!(enc, "oai:ENC");
            }
            other => panic!("expected reasoning stop block, got {other:?}"),
        }

        // 入口 encode：凭据随 reasoning item 收尾事件原样下发
        let cchunks = adapter
            .encode(&ctx, &evs[1], &mut StreamEncodeState::default())
            .unwrap();
        let raw = serde_json::to_string(
            &cchunks[0]
                .data
                .as_json()
                .expect("reasoning stop 应为 JSON 事件"),
        )
        .unwrap();
        assert!(raw.contains("encrypted_content"));
        assert!(raw.contains("ENC"));

        // 无明文且无凭据：回退 message 形态收尾，不发空 reasoning item
        let evs = adapter
            .decode(
                &ctx,
                &mut StreamDecodeState::default(),
                &chunk(json!({
                    "type": "response.output_item.done", "output_index": 0,
                    "item": { "type": "reasoning", "summary": [] }
                })),
            )
            .unwrap();
        assert!(matches!(
            &evs[0],
            CoreStreamEvent::BlockStop { block: None, .. }
        ));
    }

    /// ToolUse 收尾必须是 function_call 形态：arguments.done 先携带完整
    /// arguments，随后 output_item.done 发 function_call item（不是 message）。
    /// 历史上这里误发 message item，客户端按输出类型归类时工具调用丢失。
    #[test]
    fn encodes_tool_use_stop_as_function_call() {
        let adapter = OpenAiResponsesAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);
        let mut st = StreamEncodeState::default();

        let start = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStart {
                    index: 1,
                    block: ContentBlock::ToolUse {
                        id: "call_9".into(),
                        item_id: Some("fc_up".into()),
                        name: "get_time".into(),
                        namespace: None,
                        input: json!({}),
                        signature: None,
                    },
                },
                &mut st,
            )
            .unwrap();
        let item = &start[0].data.as_json().unwrap();
        assert_eq!(item["type"], "response.output_item.added");
        assert_eq!(item["item"]["type"], "function_call");
        // item 级 id 必填：严格校验的客户端缺它直接丢弃整个事件；
        // 上游给了 item id 时优先透传
        assert_eq!(item["item"]["id"], "fc_up");
        assert_eq!(item["item"]["call_id"], "call_9");

        adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockDelta {
                    index: 1,
                    delta: StreamDelta::ToolInput {
                        partial_json: "{\"tz\":\"UTC\"}".into(),
                    },
                },
                &mut st,
            )
            .unwrap();

        let stop = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStop {
                    index: 1,
                    block: Some(ContentBlock::ToolUse {
                        id: "call_9".into(),
                        item_id: Some("fc_up".into()),
                        name: "get_time".into(),
                        namespace: None,
                        input: json!({"tz": "UTC"}),
                        signature: None,
                    }),
                },
                &mut st,
            )
            .unwrap();
        let types: Vec<&str> = stop
            .iter()
            .filter_map(|c| c.data.as_json()?.get("type")?.as_str())
            .collect();
        assert_eq!(
            types,
            vec![
                "response.function_call_arguments.done",
                "response.output_item.done"
            ],
            "FC 收尾应是 arguments.done + output_item.done"
        );
        // arguments.done / output_item.done 的 item_id 与 item.id 一致
        assert_eq!(stop[0].data.as_json().unwrap()["item_id"], "fc_up");
        let done_item = &stop[1].data.as_json().unwrap();
        assert_eq!(done_item["item"]["type"], "function_call");
        assert_eq!(done_item["item"]["id"], "fc_up");
        assert_eq!(done_item["item"]["call_id"], "call_9");
        assert_eq!(done_item["item"]["arguments"], "{\"tz\":\"UTC\"}");
        assert_eq!(done_item["item"]["status"], "completed");
    }

    /// 推理摘要增量必须携带 summary_index——严格校验的客户端（AI SDK 系）
    /// schema 里它是必填 number，缺失会使整个增量事件被丢弃。
    #[test]
    fn reasoning_summary_delta_carries_summary_index() {
        let adapter = OpenAiResponsesAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);
        let mut st = StreamEncodeState::default();
        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockDelta {
                    index: 0,
                    delta: StreamDelta::Reasoning { text: "hmm".into() },
                },
                &mut st,
            )
            .unwrap();
        let d = evs[0].data.as_json().unwrap();
        assert_eq!(d["type"], "response.reasoning_summary_text.delta");
        assert_eq!(d["summary_index"], 0);
        assert_eq!(d["item_id"], "msg_0");
        assert_eq!(d["delta"], "hmm");
    }

    /// response.completed 必须携带组装好的 output（reasoning/message/
    /// function_call 全量 item）——客户端（如 Codex）直接用它更新会话状态。
    #[test]
    fn completed_carries_assembled_output() {
        let adapter = OpenAiResponsesAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);
        let mut st = StreamEncodeState::default();

        let _ = adapter.encode(
            &ctx,
            &CoreStreamEvent::MessageStart {
                id: "resp_x".into(),
                model: "gpt-x".into(),
            },
            &mut st,
        );
        let _ = adapter.encode(
            &ctx,
            &CoreStreamEvent::BlockStart {
                index: 0,
                block: ContentBlock::Reasoning {
                    text: "t".into(),
                    signature: Some("oai:ENC".into()),
                    redacted: false,
                },
            },
            &mut st,
        );
        let _ = adapter.encode(
            &ctx,
            &CoreStreamEvent::BlockStop {
                index: 0,
                block: Some(ContentBlock::Reasoning {
                    text: "t".into(),
                    signature: Some("oai:ENC".into()),
                    redacted: false,
                }),
            },
            &mut st,
        );
        let _ = adapter.encode(
            &ctx,
            &CoreStreamEvent::BlockStart {
                index: 1,
                block: ContentBlock::text(""),
            },
            &mut st,
        );
        let _ = adapter.encode(
            &ctx,
            &CoreStreamEvent::BlockDelta {
                index: 1,
                delta: StreamDelta::Text { text: "Hi".into() },
            },
            &mut st,
        );
        let _ = adapter.encode(
            &ctx,
            &CoreStreamEvent::BlockStop {
                index: 1,
                block: Some(ContentBlock::text("Hi")),
            },
            &mut st,
        );

        let done = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::MessageDelta {
                    stop_reason: Some(StopReason::EndTurn),
                    usage: None,
                },
                &mut st,
            )
            .unwrap();
        let completed = done.iter().find_map(|c| c.data.as_json().cloned()).unwrap();
        // 上游未上报 usage 时也必须发 usage 对象（必填字段）
        assert_eq!(completed["response"]["usage"]["input_tokens"], 0);
        let output = completed["response"]["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(
            output[0]["encrypted_content"], "ENC",
            "本家凭据出站解除标记"
        );
        assert_eq!(output[1]["type"], "message");
        assert_eq!(output[1]["content"][0]["text"], "Hi");
    }

    /// 回归（cctq/new-api 实测）：独立 `error` 事件帧携带真实失败原因
    /// （rate_limit_error/turn_rate_limit_exceeded + Retry-After），须透出为
    /// Core Error 事件而非丢弃——code 与 retry-after 一并进入消息文本。
    #[test]
    fn decodes_inband_error_event_with_retry_after() {
        let adapter = OpenAiResponsesAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);
        let mut st = StreamDecodeState::default();
        let evs = adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({
                    "type": "error",
                    "status": 429,
                    "headers": {"Retry-After": "53"},
                    "error": {
                        "type": "rate_limit_error",
                        "code": "turn_rate_limit_exceeded",
                        "message": "Codex is at capacity, please try again later"
                    }
                })),
            )
            .unwrap();
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            CoreStreamEvent::Error { message } => {
                assert_eq!(
                    message,
                    "turn_rate_limit_exceeded: Codex is at capacity, please try again later [retry-after=53s]"
                );
            }
            _ => panic!("expected error event: {evs:?}"),
        }
    }

    /// error 帧缺 code 时只透出 message；无 message 时不产事件。
    #[test]
    fn decodes_inband_error_minimal() {
        let adapter = OpenAiResponsesAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);
        let mut st = StreamDecodeState::default();
        let evs = adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({"type": "error", "error": {"message": "boom"}})),
            )
            .unwrap();
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            CoreStreamEvent::Error { message } => assert_eq!(message, "boom"),
            _ => panic!("expected error event: {evs:?}"),
        }
        let evs = adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({"type": "error", "error": {"code": "x"}})),
            )
            .unwrap();
        assert!(evs.is_empty(), "无 message 的 error 帧不产事件: {evs:?}");
    }
}
