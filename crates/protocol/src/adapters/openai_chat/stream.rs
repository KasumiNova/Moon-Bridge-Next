//! OpenAI Chat Completions 流式：Core 事件 ↔ Chat SSE chunk。
//!
//! Chat SSE 形如 `data: {"choices":[{"delta":{...}}]}`，以 `data: [DONE]` 结束。
//! encode 供 Chat 作入口时输出；decode 供 Chat 作上游时解析。
//!
//! Chat 流本身**没有块边界事件**：正文/推理增量按 delta 字段区分，
//! tool_calls 按 `index`（工具调用序数）区分。decode 侧负责把这些
//! 合成 Core 的 BlockStart/BlockDelta/BlockStop——下游客户端协议
//! （Anthropic/Responses）依赖块边界驱动自己的状态机，缺了边界事件
//! 它们的块永远不收尾。

use async_trait::async_trait;
use moonbridge_core::{ContentBlock, CoreStreamEvent, Protocol, Result, StreamDelta, Usage};
use serde_json::{json, Value};

use super::dto::{map_finish_reason, unmap_stop_reason, usage_from_chat, usage_object};
use super::OpenAiChatAdapter;
use crate::adapter::{
    ClientStreamAdapter, ProviderStreamAdapter, StreamDecodeState, StreamEncodeState,
};
use crate::context::ReqCtx;
use crate::raw::{ChunkStage, RawBody, RawChunk};

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 构造一个 Chat SSE chunk（客户端方向）。id 用 MessageStart 记录下的
/// 真实响应 id（上游未发时退化为 `chatcmpl-{request_id}`）。
fn chat_chunk(st: &StreamEncodeState, ctx: &ReqCtx, delta: Value, finish: Value) -> RawChunk {
    let id = if st.message_id.is_empty() {
        format!("chatcmpl-{}", ctx.request_id)
    } else {
        st.message_id.clone()
    };
    RawChunk::json(
        ChunkStage::ClientChunk,
        Protocol::OpenAiChat,
        None,
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": now_unix(),
            "model": ctx.model_alias,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        }),
    )
}

/// 构造带 tool_calls delta 的 chunk。`ordinal` 为工具调用序数
/// （与 Core 块索引区分：tool_calls[].index 只在调用内计数）。
fn tool_chunk(st: &StreamEncodeState, ctx: &ReqCtx, tc: Value) -> RawChunk {
    chat_chunk(st, ctx, json!({ "tool_calls": [tc] }), Value::Null)
}

/// `data: [DONE]` 终止标记（非 JSON，用 Text body）。
fn done_chunk() -> RawChunk {
    RawChunk {
        stage: ChunkStage::ClientChunk,
        protocol: Protocol::OpenAiChat,
        provider: None,
        event: None,
        data: RawBody::Text {
            text: "[DONE]".to_string(),
        },
        raw: String::new(),
    }
}

#[async_trait]
impl ClientStreamAdapter for OpenAiChatAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::OpenAiChat
    }

    fn encode(
        &self,
        ctx: &ReqCtx,
        ev: &CoreStreamEvent,
        st: &mut StreamEncodeState,
    ) -> Result<Vec<RawChunk>> {
        let mut out = Vec::new();
        match ev {
            CoreStreamEvent::MessageStart { id, .. } => {
                st.message_id = id.clone();
                out.push(chat_chunk(
                    st,
                    ctx,
                    json!({ "role": "assistant", "content": "" }),
                    Value::Null,
                ));
            }
            CoreStreamEvent::BlockStart { index, block } => {
                st.note_start(*index, block);
                if let ContentBlock::ToolUse { id, name, .. } = block {
                    let ord = st.tool_ordinal(*index);
                    out.push(tool_chunk(
                        st,
                        ctx,
                        json!({
                            "index": ord, "id": id, "type": "function",
                            "function": { "name": name, "arguments": "" },
                        }),
                    ));
                }
            }
            CoreStreamEvent::BlockDelta { index, delta } => match delta {
                StreamDelta::Text { text } => {
                    st.push_delta(*index, text);
                    out.push(chat_chunk(st, ctx, json!({ "content": text }), Value::Null));
                }
                StreamDelta::ToolInput { partial_json } => {
                    st.push_delta(*index, partial_json);
                    let ord = st.tool_ordinal(*index);
                    out.push(tool_chunk(
                        st,
                        ctx,
                        json!({ "index": ord, "function": { "arguments": partial_json } }),
                    ));
                }
                // 双写两种生态约定：DeepSeek（reasoning_content）与 OpenRouter/vLLM（reasoning）
                StreamDelta::Reasoning { text } => {
                    st.push_delta(*index, text);
                    out.push(chat_chunk(
                        st,
                        ctx,
                        json!({ "reasoning_content": text, "reasoning": text }),
                        Value::Null,
                    ));
                }
                // 凭据以 <mb-cot> 定界标记追加：客户端累积 reasoning_content 后
                // 标记自然位于尾部，下一轮历史回传时由 split_mb_cot 解析还原
                StreamDelta::ReasoningSignature { signature } => {
                    let marker = format!("<mb-cot>{signature}</mb-cot>");
                    out.push(chat_chunk(
                        st,
                        ctx,
                        json!({ "reasoning_content": marker, "reasoning": marker }),
                        Value::Null,
                    ));
                }
            },
            CoreStreamEvent::BlockStop { index, block } => {
                // 工具块的 arguments 可能只在收尾完整块里到达（Responses 上游
                // `arguments.done` 单帧形态）：若本块从未下发过非空 arguments
                // 增量，补发一帧携带完整 input 的 arguments delta——否则客户端
                // 收到 `arguments: ""` 的空参数工具调用（参数静默丢失）。
                let input = match block.as_ref() {
                    Some(ContentBlock::ToolUse { input, .. }) => Some(input),
                    _ => None,
                };
                let is_tool = input.is_some()
                    || matches!(st.blocks.get(index), Some(ContentBlock::ToolUse { .. }));
                if is_tool {
                    let emitted = st.delta_acc.get(index).is_some_and(|s| !s.is_empty());
                    if !emitted {
                        let args = input
                            .filter(|v| !v.is_null())
                            .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "{}".into()))
                            .unwrap_or_else(|| "{}".to_string());
                        let ord = st.tool_ordinal(*index);
                        out.push(tool_chunk(
                            st,
                            ctx,
                            json!({ "index": ord, "function": { "arguments": args } }),
                        ));
                    }
                    return Ok(out);
                }
                // Text/Reasoning 同理：整段内容只在收尾块到达时补发对应增量。
                let text_field = match block.as_ref() {
                    Some(ContentBlock::Text { text }) => Some(("content", text)),
                    Some(ContentBlock::Reasoning { text, .. }) => Some(("reasoning", text)),
                    _ => None,
                };
                if let Some((field, text)) = text_field {
                    let emitted = st.delta_acc.get(index).is_some_and(|s| !s.is_empty());
                    if !emitted && !text.is_empty() {
                        let delta = if field == "reasoning" {
                            json!({ "reasoning_content": text, "reasoning": text })
                        } else {
                            json!({ "content": text })
                        };
                        out.push(chat_chunk(st, ctx, delta, Value::Null));
                    }
                }
            }
            CoreStreamEvent::MessageDelta { stop_reason, usage } => {
                if let Some(u) = usage {
                    st.merge_usage(u);
                }
                // stop_reason 未知（上游中途 usage 更新）时 finish_reason 必须为
                // null——伪造 "stop" 会让客户端在流中段误判回合结束。
                let finish = match stop_reason {
                    Some(r) => json!(unmap_stop_reason(Some(*r))),
                    None => Value::Null,
                };
                let mut c = chat_chunk(st, ctx, json!({}), finish);
                if st.usage_acc != Usage::default() {
                    if let RawBody::Json { value } = &mut c.data {
                        value["usage"] = usage_object(&st.usage_acc);
                    }
                }
                out.push(c);
            }
            CoreStreamEvent::MessageStop => out.push(done_chunk()),
            CoreStreamEvent::Error { message } => {
                out.push(RawChunk::json(
                    ChunkStage::ClientChunk,
                    Protocol::OpenAiChat,
                    None,
                    json!({ "error": { "message": message } }),
                ));
            }
        }
        Ok(out)
    }
}

/// 从 chunk 取出 JSON data（[DONE] 等非 JSON 返回 None）。
fn chunk_json(chunk: &RawChunk) -> Option<Value> {
    if let Some(v) = chunk.data.as_json() {
        return Some(v.clone());
    }
    serde_json::from_str::<Value>(&chunk.raw).ok()
}

/// 关闭 `st` 中所有仍开启的块，按开启顺序补发 BlockStop（携带累积出的
/// 完整块：ToolUse 带上拼好的 arguments）。上游到达 finish_reason 或
/// [DONE] 时调用——没有这一步，下游客户端的块状态机永远等不到收尾。
fn close_open_blocks(st: &mut StreamDecodeState) -> Vec<CoreStreamEvent> {
    let open = std::mem::take(&mut st.open_blocks);
    let mut out = Vec::with_capacity(open.len());
    for index in open {
        // ToolUse 的参数在 args_acc 里拼成了完整 JSON，回填进 input
        if let Some(args) = st.args_acc.remove(&index) {
            if let Some(ContentBlock::ToolUse { input, .. }) = st.blocks.get_mut(&index) {
                *input = serde_json::from_str(&args).unwrap_or_else(|_| json!({"_raw": args}));
            }
        }
        // Chat 上游无独立凭据字段：推理明文本身即回传凭据（thinking 模式
        // 上游要求 reasoning_content 随历史带回，缺失 400）。收尾时给
        // 无凭据推理打 chat: 自凭据标记——客户端方向据此下发可回传凭据，
        // 异源上游按外源凭据降级。流式增量不含 <mb-cot> 解析（delta 已
        // 下发不可回改），故整条 reasoning_content 打标记而非拆凭据。
        if let Some(ContentBlock::Reasoning {
            text, signature, ..
        }) = st.blocks.get_mut(&index)
        {
            if signature.is_none() && !text.is_empty() {
                *signature = Some(format!("{}{}", crate::adapters::SIG_CHAT, text));
            }
        }
        out.push(CoreStreamEvent::BlockStop {
            index,
            block: st.blocks.get(&index).cloned(),
        });
    }
    out
}

#[async_trait]
impl ProviderStreamAdapter for OpenAiChatAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::OpenAiChat
    }

    fn decode(
        &self,
        _ctx: &ReqCtx,
        st: &mut StreamDecodeState,
        chunk: &RawChunk,
    ) -> Result<Vec<CoreStreamEvent>> {
        // [DONE] 终止标记：先把仍开启的块收尾，再发 MessageStop
        if let RawBody::Text { text } = &chunk.data {
            if text.trim() == "[DONE]" {
                let mut out = close_open_blocks(st);
                out.push(CoreStreamEvent::MessageStop);
                return Ok(out);
            }
        }
        let Some(data) = chunk_json(chunk) else {
            return Ok(Vec::new());
        };
        // 带内错误帧：`{"error":{...}}`（new-api 类中转的形态）——取 code 与
        // message 合成错误文本，透出而非吞掉。
        if let Some(err) = data.get("error") {
            let code = err.get("code").and_then(|v| v.as_str()).unwrap_or_default();
            let message = err
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            if !message.is_empty() {
                let text = if code.is_empty() {
                    message.to_string()
                } else {
                    format!("{code}: {message}")
                };
                return Ok(vec![CoreStreamEvent::Error { message: text }]);
            }
        }
        let mut out = Vec::new();
        let choice = data
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .cloned()
            .unwrap_or(Value::Null);
        let delta = choice.get("delta").cloned().unwrap_or(Value::Null);

        // 首个含实际内容（role/正文/推理/工具调用任一）的 chunk 视为消息开始；
        // 有些上游只发一次 role，有些每次发，故用 message_started 去重。
        let has_content = delta.get("role").is_some()
            || delta.get("content").is_some()
            || delta.get("reasoning_content").is_some()
            || delta.get("reasoning").is_some()
            || delta.get("tool_calls").is_some();
        if !st.message_started && has_content {
            st.message_started = true;
            let id = data
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let model = data
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            out.push(CoreStreamEvent::MessageStart { id, model });
        }

        // 推理增量：DeepSeek 系 reasoning_content / OpenRouter 系 reasoning。
        // 独占 "reasoning" 槽位——与正文同 index 会让下游把 text_delta 混进
        // thinking 块（协议违规）。
        if let Some(text) = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(|v| v.as_str())
        {
            if !text.is_empty() {
                let (idx, is_new) = st.slot("reasoning");
                if is_new {
                    st.open_block(idx);
                    out.push(CoreStreamEvent::BlockStart {
                        index: idx,
                        block: ContentBlock::Reasoning {
                            text: String::new(),
                            signature: None,
                            redacted: false,
                        },
                    });
                }
                st.push_text(idx, true, text);
                out.push(CoreStreamEvent::BlockDelta {
                    index: idx,
                    delta: StreamDelta::Reasoning {
                        text: text.to_string(),
                    },
                });
            }
        }

        if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
            if !text.is_empty() {
                let (idx, is_new) = st.slot("text");
                if is_new {
                    st.open_block(idx);
                    out.push(CoreStreamEvent::BlockStart {
                        index: idx,
                        block: ContentBlock::text(String::new()),
                    });
                }
                st.push_text(idx, false, text);
                out.push(CoreStreamEvent::BlockDelta {
                    index: idx,
                    delta: StreamDelta::Text {
                        text: text.to_string(),
                    },
                });
            }
        }

        if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tcs {
                // 上游 index 是工具调用序数，与正文槽位无关——映射到独立
                // 块索引，避免与 text/reasoning 冲突。
                let tc_index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                let (index, is_new) = st.slot(&format!("tc:{tc_index}"));
                let fn_obj = tc.get("function").cloned().unwrap_or(Value::Null);
                if is_new {
                    // Chat 流按序下发工具调用：新序数出现即代表上一个
                    // 工具块已完结，先补发它的 BlockStop。
                    let pending: Vec<usize> = st
                        .open_blocks
                        .iter()
                        .copied()
                        .filter(|&i| {
                            matches!(st.blocks.get(&i), Some(ContentBlock::ToolUse { .. }))
                        })
                        .collect();
                    for i in pending {
                        st.close_block(i);
                        if let Some(args) = st.args_acc.remove(&i) {
                            if let Some(ContentBlock::ToolUse { input, .. }) = st.blocks.get_mut(&i)
                            {
                                *input = serde_json::from_str(&args)
                                    .unwrap_or_else(|_| json!({"_raw": args}));
                            }
                        }
                        out.push(CoreStreamEvent::BlockStop {
                            index: i,
                            block: st.blocks.get(&i).cloned(),
                        });
                    }
                }
                // 块尚未起步（首个 tc chunk，或此前只收到过参数增量）→
                // 建 ToolUse 并 BlockStart。上游缺 id 时同样开块——否则该
                // 调用的参数增量没有归属块。
                let started = matches!(st.blocks.get(&index), Some(ContentBlock::ToolUse { .. }));
                if !started {
                    let block = ContentBlock::ToolUse {
                        id: tc
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        item_id: None,
                        name: fn_obj
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        namespace: None,
                        input: json!({}),
                        signature: None,
                    };
                    st.blocks.insert(index, block.clone());
                    st.open_block(index);
                    out.push(CoreStreamEvent::BlockStart { index, block });
                }
                if let Some(args) = fn_obj.get("arguments").and_then(|v| v.as_str()) {
                    if !args.is_empty() {
                        st.args_acc.entry(index).or_default().push_str(args);
                        out.push(CoreStreamEvent::BlockDelta {
                            index,
                            delta: StreamDelta::ToolInput {
                                partial_json: args.to_string(),
                            },
                        });
                    }
                }
            }
        }

        if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            // finish 到来意味着所有块完结——先收尾再报 MessageDelta
            out.extend(close_open_blocks(st));
            let usage = data.get("usage").map(usage_from_chat);
            out.push(CoreStreamEvent::MessageDelta {
                stop_reason: map_finish_reason(fr),
                usage,
            });
        } else if let Some(u) = data.get("usage") {
            // usage 专用终块（上游 stream_options.include_usage 的产物：choices 为
            // 空、无 finish_reason）：无 stop_reason，仅作中途用量上报。
            out.push(CoreStreamEvent::MessageDelta {
                stop_reason: None,
                usage: Some(usage_from_chat(u)),
            });
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(v: Value) -> RawChunk {
        RawChunk::json(ChunkStage::UpstreamChunk, Protocol::OpenAiChat, None, v)
    }

    #[test]
    fn encodes_text_delta_and_done() {
        let adapter = OpenAiChatAdapter;
        let mut ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        ctx.model_alias = "gpt-4o".to_string();

        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockDelta {
                    index: 0,
                    delta: StreamDelta::Text { text: "Hi".into() },
                },
                &mut StreamEncodeState::default(),
            )
            .unwrap();
        assert_eq!(evs.len(), 1);
        assert_eq!(
            evs[0].data.as_json().unwrap()["choices"][0]["delta"]["content"],
            "Hi"
        );

        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::MessageStop,
                &mut StreamEncodeState::default(),
            )
            .unwrap();
        match &evs[0].data {
            RawBody::Text { text } => assert_eq!(text, "[DONE]"),
            other => panic!("expected [DONE] text, got {other:?}"),
        }
    }

    #[test]
    fn encodes_reasoning_delta_with_both_conventions() {
        let adapter = OpenAiChatAdapter;
        let mut ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        ctx.model_alias = "m".to_string();
        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockDelta {
                    index: 0,
                    delta: StreamDelta::Reasoning {
                        text: "think".into(),
                    },
                },
                &mut StreamEncodeState::default(),
            )
            .unwrap();
        let d = &evs[0].data.as_json().unwrap()["choices"][0]["delta"];
        assert_eq!(d["reasoning_content"], "think", "DeepSeek 约定");
        assert_eq!(d["reasoning"], "think", "OpenRouter/vLLM 约定");
    }

    /// tool_calls[].index 必须是工具调用序数，不能直接用 Core 块索引——
    /// 前置文本块占位后会产生稀疏序号，OpenAI SDK 按 index 累积出带空洞数组。
    #[test]
    fn encode_renumbers_tool_call_index() {
        let adapter = OpenAiChatAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        let mut st = StreamEncodeState::default();

        // Core 块索引 2 处的工具调用应编码为 tool_calls index 0
        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStart {
                    index: 2,
                    block: ContentBlock::ToolUse {
                        id: "call_1".into(),
                        item_id: None,
                        name: "get_time".into(),
                        namespace: None,
                        input: json!({}),
                        signature: None,
                    },
                },
                &mut st,
            )
            .unwrap();
        assert_eq!(
            evs[0].data.as_json().unwrap()["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );

        // 同一块的后续参数增量沿用同一序数
        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockDelta {
                    index: 2,
                    delta: StreamDelta::ToolInput {
                        partial_json: "{\"a\":".into(),
                    },
                },
                &mut st,
            )
            .unwrap();
        assert_eq!(
            evs[0].data.as_json().unwrap()["choices"][0]["delta"]["tool_calls"][0]["index"],
            0
        );

        // 第二个工具（Core 索引 5）→ 序数 1
        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStart {
                    index: 5,
                    block: ContentBlock::ToolUse {
                        id: "call_2".into(),
                        item_id: None,
                        name: "f".into(),
                        namespace: None,
                        input: json!({}),
                        signature: None,
                    },
                },
                &mut st,
            )
            .unwrap();
        assert_eq!(
            evs[0].data.as_json().unwrap()["choices"][0]["delta"]["tool_calls"][0]["index"],
            1
        );
    }

    #[test]
    fn decodes_reasoning_content_from_upstream() {
        let adapter = OpenAiChatAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        for key in ["reasoning_content", "reasoning"] {
            let evs = adapter
                .decode(
                    &ctx,
                    &mut StreamDecodeState::default(),
                    &chunk(json!({"choices":[{"index":0,"delta":{key: "hmm"}}]})),
                )
                .unwrap();
            let has_reasoning = evs.iter().any(|e| {
                matches!(e, CoreStreamEvent::BlockDelta { delta: StreamDelta::Reasoning { text }, .. } if text == "hmm")
            });
            assert!(has_reasoning, "{key} 应解码为推理增量: {evs:?}");
        }
    }

    #[test]
    fn decodes_chat_stream() {
        let adapter = OpenAiChatAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);

        let evs = adapter
            .decode(&ctx, &mut StreamDecodeState::default(), &chunk(json!({"id":"c1","model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]})))
            .unwrap();
        assert!(matches!(evs[0], CoreStreamEvent::MessageStart { .. }));
        // 正文块现在带 BlockStart
        assert!(matches!(evs[1], CoreStreamEvent::BlockStart { .. }));
        match &evs[2] {
            CoreStreamEvent::BlockDelta {
                delta: StreamDelta::Text { text },
                ..
            } => assert_eq!(text, "Hello"),
            other => panic!("expected text delta, got {other:?}"),
        }

        let mut st = StreamDecodeState {
            message_started: true,
            ..Default::default()
        };
        let evs = adapter
            .decode(&ctx, &mut st, &chunk(json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}})))
            .unwrap();
        assert!(evs
            .iter()
            .any(|e| matches!(e, CoreStreamEvent::MessageDelta { .. })));

        // [DONE]
        let done = RawChunk {
            stage: ChunkStage::UpstreamChunk,
            protocol: Protocol::OpenAiChat,
            provider: None,
            event: None,
            data: RawBody::Text {
                text: "[DONE]".into(),
            },
            raw: String::new(),
        };
        let evs = adapter.decode(&ctx, &mut st, &done).unwrap();
        assert!(evs
            .iter()
            .any(|e| matches!(e, CoreStreamEvent::MessageStop)));
    }

    /// 回归：收尾的 reasoning 块必须打 chat: 自凭据——Responses 等客户端
    /// 只回传带凭据的 reasoning item，thinking 模式上游缺 reasoning_content
    /// 会 400；凭据 payload 为推理原文（chat 上游无独立凭据字段）。
    #[test]
    fn reasoning_block_close_carries_chat_credential() {
        let adapter = OpenAiChatAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        let mut st = StreamDecodeState::default();

        adapter
            .decode(&ctx, &mut st, &chunk(json!({"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think"}}]})))
            .unwrap();
        let evs = adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})),
            )
            .unwrap();
        let stop = evs.iter().find(|e| {
            matches!(
                e,
                CoreStreamEvent::BlockStop {
                    block: Some(ContentBlock::Reasoning { .. }),
                    ..
                }
            )
        });
        let Some(CoreStreamEvent::BlockStop {
            block: Some(ContentBlock::Reasoning {
                text, signature, ..
            }),
            ..
        }) = stop
        else {
            panic!("reasoning BlockStop 应携带完整块: {evs:?}")
        };
        assert_eq!(text, "think");
        assert_eq!(
            signature.as_deref(),
            Some("chat:think"),
            "无凭据推理收尾时应打 chat: 自凭据"
        );
    }

    /// 回归：reasoning_content 与 content 必须落在**不同**块索引——
    /// 旧实现都写 index 0，Anthropic 客户端会把 text_delta 混进 thinking 块。
    #[test]
    fn reasoning_and_text_use_distinct_block_indexes() {
        let adapter = OpenAiChatAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        let mut st = StreamDecodeState::default();

        let evs = adapter
            .decode(&ctx, &mut st, &chunk(json!({"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think"}}]})))
            .unwrap();
        let r_idx = evs
            .iter()
            .find_map(|e| match e {
                CoreStreamEvent::BlockDelta {
                    index,
                    delta: StreamDelta::Reasoning { .. },
                } => Some(*index),
                _ => None,
            })
            .expect("应有推理增量");

        let evs = adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({"choices":[{"index":0,"delta":{"content":"visible"}}]})),
            )
            .unwrap();
        let t_idx = evs
            .iter()
            .find_map(|e| match e {
                CoreStreamEvent::BlockDelta {
                    index,
                    delta: StreamDelta::Text { .. },
                } => Some(*index),
                _ => None,
            })
            .expect("应有文本增量");
        assert_ne!(r_idx, t_idx, "推理与正文不得共享块索引");
    }

    /// 回归：tool_calls 的上游 index 是工具序数而非内容块序数——不得与
    /// 正文 index 0 冲突；且块要开也要收（finish 时统一收尾）。
    #[test]
    fn tool_calls_get_own_indexes_and_close_at_finish() {
        let adapter = OpenAiChatAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        let mut st = StreamDecodeState::default();

        // 正文在块 0
        adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(
                    json!({"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"}}]}),
                ),
            )
            .unwrap();
        // 工具调用 tc.index=0 → 不得落在块 0
        let evs = adapter
            .decode(&ctx, &mut st, &chunk(json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"f","arguments":"{\"a\":1}"}}]}}]})))
            .unwrap();
        let tc_idx = evs
            .iter()
            .find_map(|e| match e {
                CoreStreamEvent::BlockStart {
                    index,
                    block: ContentBlock::ToolUse { .. },
                } => Some(*index),
                _ => None,
            })
            .expect("应有 ToolUse BlockStart");
        assert_ne!(tc_idx, 0, "tool_call 不得与正文块撞索引");

        // finish_reason → 全部块收尾 + MessageDelta
        let evs = adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]})),
            )
            .unwrap();
        let stops: Vec<usize> = evs
            .iter()
            .filter_map(|e| match e {
                CoreStreamEvent::BlockStop { index, .. } => Some(*index),
                _ => None,
            })
            .collect();
        assert!(
            stops.contains(&0) && stops.contains(&tc_idx),
            "两个块都应收尾: {stops:?}"
        );
        // 收尾的 ToolUse 应带回完整参数
        let tool_stop = evs
            .iter()
            .find(|e| matches!(e, CoreStreamEvent::BlockStop { index, .. } if *index == tc_idx));
        if let Some(CoreStreamEvent::BlockStop {
            block: Some(ContentBlock::ToolUse { input, .. }),
            ..
        }) = tool_stop
        {
            assert_eq!(input["a"], 1, "arguments 应拼合成完整 JSON: {input:?}");
        } else {
            panic!("ToolUse BlockStop 应携带完整块: {tool_stop:?}");
        }
        assert!(evs.iter().any(|e| matches!(
            e,
            CoreStreamEvent::MessageDelta {
                stop_reason: Some(_),
                ..
            }
        )));
    }

    /// 回归：工具 arguments 只在收尾完整块里到达（上游不发增量帧）时，
    /// BlockStop 必须补发携带完整 input 的 arguments delta——否则下游拿到
    /// `arguments: ""` 的空参数调用，参数静默丢失。
    #[test]
    fn encodes_stop_with_args_when_no_input_delta() {
        let adapter = OpenAiChatAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        let mut st = StreamEncodeState::default();
        let block = ContentBlock::ToolUse {
            id: "call_1".into(),
            item_id: None,
            name: "read_file".into(),
            namespace: None,
            input: json!({"path": "/tmp/x"}),
            signature: None,
        };
        adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStart {
                    index: 0,
                    block: block.clone(),
                },
                &mut st,
            )
            .unwrap();
        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStop {
                    index: 0,
                    block: Some(block),
                },
                &mut st,
            )
            .unwrap();
        assert_eq!(evs.len(), 1, "应补发一帧 arguments delta: {evs:?}");
        let tc = &evs[0].data.as_json().unwrap()["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["index"], 0);
        assert_eq!(tc["function"]["arguments"], r#"{"path":"/tmp/x"}"#);
    }

    /// 已有非空参数增量时 BlockStop 不再补发。
    #[test]
    fn no_args_chunk_when_delta_streamed() {
        let adapter = OpenAiChatAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        let mut st = StreamEncodeState::default();
        let block = ContentBlock::ToolUse {
            id: "call_2".into(),
            item_id: None,
            name: "read_file".into(),
            namespace: None,
            input: Value::Null,
            signature: None,
        };
        adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStart { index: 0, block },
                &mut st,
            )
            .unwrap();
        adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockDelta {
                    index: 0,
                    delta: StreamDelta::ToolInput {
                        partial_json: r#"{"a":1}"#.into(),
                    },
                },
                &mut st,
            )
            .unwrap();
        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStop {
                    index: 0,
                    block: None,
                },
                &mut st,
            )
            .unwrap();
        assert!(evs.is_empty(), "已下发参数增量时收尾应静默: {evs:?}");
    }
}
