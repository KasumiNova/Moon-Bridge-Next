//! Anthropic 上游 Adapter —— 流式解码。
//!
//! 把 Anthropic Messages 的 SSE chunk 解码为协议中立的 [`CoreStreamEvent`]。
//! Anthropic 的流事件模型与 Core IR 高度一致（Core IR 即参考其设计），因此映射
//! 直接：`message_start`→MessageStart、`content_block_*`→Block*、
//! `message_delta`→MessageDelta、`message_stop`→MessageStop。

use async_trait::async_trait;
use moonbridge_core::{
    ContentBlock, CoreStreamEvent, Protocol, Result, StopReason, StreamDelta, Usage,
};
use serde_json::{json, Value};

use super::provider::{anthropic_usage_out, map_stop_reason, unmap_stop_reason};
use super::AnthropicAdapter;
use crate::adapter::{
    ClientStreamAdapter, ProviderStreamAdapter, StreamDecodeState, StreamEncodeState,
};
use crate::context::ReqCtx;
use crate::raw::{ChunkStage, RawChunk};

fn stop_reason(s: &str) -> Option<StopReason> {
    map_stop_reason(s)
}

fn u32_of(v: &Value, key: &str) -> u32 {
    v.get(key).and_then(|x| x.as_u64()).unwrap_or(0) as u32
}

/// 从 chunk 取出 JSON data：优先已解析的 data，回退解析 raw 文本。
fn chunk_json(chunk: &RawChunk) -> Option<Value> {
    if let Some(v) = chunk.data.as_json() {
        return Some(v.clone());
    }
    serde_json::from_str::<Value>(&chunk.raw).ok()
}

#[async_trait]
impl ProviderStreamAdapter for AnthropicAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::Anthropic
    }

    fn decode(
        &self,
        _ctx: &ReqCtx,
        _st: &mut StreamDecodeState,
        chunk: &RawChunk,
    ) -> Result<Vec<CoreStreamEvent>> {
        let data = match chunk_json(chunk) {
            Some(d) => d,
            None => return Ok(Vec::new()),
        };
        let ty = data.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let mut out = Vec::new();

        match ty {
            "message_start" => {
                let msg = data.get("message").cloned().unwrap_or_else(|| Value::Null);
                let id = msg
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let model = msg
                    .get("model")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                out.push(CoreStreamEvent::MessageStart { id, model });
                if let Some(u) = msg.get("usage") {
                    // 中途 usage 更新（非终态）：stop_reason=None 表示 stop_reason 未知，
                    // 各入口 encode 不得据此伪造收尾语义。input 并入缓存读写，与非流式
                    // to_core_response 的 Core 口径（prompt 总量含缓存）一致。
                    let cache_r = u32_of(u, "cache_read_input_tokens");
                    let cache_w = u32_of(u, "cache_creation_input_tokens");
                    out.push(CoreStreamEvent::MessageDelta {
                        stop_reason: None,
                        usage: Some(Usage {
                            input_tokens: u32_of(u, "input_tokens") + cache_r + cache_w,
                            output_tokens: u32_of(u, "output_tokens"),
                            cache_read_tokens: cache_r,
                            cache_write_tokens: cache_w,
                            reasoning_tokens: 0,
                        }),
                    });
                }
            }
            "content_block_start" => {
                let index = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let cb = data
                    .get("content_block")
                    .cloned()
                    .unwrap_or_else(|| Value::Null);
                let block = match cb.get("type").and_then(|t| t.as_str()) {
                    Some("tool_use") => ContentBlock::ToolUse {
                        id: cb
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        item_id: None,
                        name: cb
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        namespace: None,
                        input: cb.get("input").cloned().unwrap_or_else(|| Value::Null),
                        signature: None,
                    },
                    Some("thinking") => ContentBlock::Reasoning {
                        text: cb
                            .get("thinking")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string(),
                        signature: None,
                        redacted: false,
                    },
                    // redacted_thinking 流式形态：content_block_start 直接携带完整
                    // data，无后续增量，立即收尾
                    Some("redacted_thinking") => ContentBlock::Reasoning {
                        text: String::new(),
                        signature: crate::adapters::tag_signature(
                            crate::adapters::SIG_ANTHROPIC,
                            cb.get("data").and_then(|v| v.as_str()).map(String::from),
                        ),
                        redacted: true,
                    },
                    _ => ContentBlock::text(
                        cb.get("text").and_then(|v| v.as_str()).unwrap_or_default(),
                    ),
                };
                // redacted 完整块的凭据转为凭据增量流出：入口 encode 的惰性开块
                // 逻辑据它以 redacted_thinking 形态还原给客户端（否则整块丢失）
                let redacted_sig = if let ContentBlock::Reasoning {
                    signature: Some(sig),
                    redacted: true,
                    ..
                } = &block
                {
                    Some(sig.clone())
                } else {
                    None
                };
                out.push(CoreStreamEvent::BlockStart { index, block });
                if let Some(sig) = redacted_sig {
                    out.push(CoreStreamEvent::BlockDelta {
                        index,
                        delta: StreamDelta::ReasoningSignature { signature: sig },
                    });
                }
            }
            "content_block_delta" => {
                let index = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let delta = data.get("delta").cloned().unwrap_or_else(|| Value::Null);
                match delta.get("type").and_then(|t| t.as_str()) {
                    Some("text_delta") => {
                        let text = delta
                            .get("text")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();
                        out.push(CoreStreamEvent::BlockDelta {
                            index,
                            delta: StreamDelta::Text {
                                text: text.to_string(),
                            },
                        });
                    }
                    Some("input_json_delta") => {
                        let partial = delta
                            .get("partial_json")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();
                        out.push(CoreStreamEvent::BlockDelta {
                            index,
                            delta: StreamDelta::ToolInput {
                                partial_json: partial.to_string(),
                            },
                        });
                    }
                    Some("thinking_delta") => {
                        let thinking = delta
                            .get("thinking")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();
                        out.push(CoreStreamEvent::BlockDelta {
                            index,
                            delta: StreamDelta::Reasoning {
                                text: thinking.to_string(),
                            },
                        });
                    }
                    // 加密 CoT 回传凭据：必须在 content_block_stop 前到达客户端，
                    // 否则多轮 thinking+tool_use 回传缺失 signature 会 400
                    Some("signature_delta") => {
                        let sig = delta
                            .get("signature")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();
                        if let Some(sig) = crate::adapters::tag_signature(
                            crate::adapters::SIG_ANTHROPIC,
                            Some(sig.to_string()),
                        ) {
                            out.push(CoreStreamEvent::BlockDelta {
                                index,
                                delta: StreamDelta::ReasoningSignature { signature: sig },
                            });
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = data.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                out.push(CoreStreamEvent::BlockStop { index, block: None });
            }
            "message_delta" => {
                let sr = data
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(|v| v.as_str())
                    .and_then(stop_reason);
                // message_delta 的 usage 不只 output_tokens：server tool（web search
                // 等）会让 input/cache 在轮内增长并在此处上报。全字段按 message_start
                // 同口径归一化读出，聚合端按字段取 max，新旧值自然取大者。
                let usage = data.get("usage").map(|u| {
                    let cache_r = u32_of(u, "cache_read_input_tokens");
                    let cache_w = u32_of(u, "cache_creation_input_tokens");
                    Usage {
                        input_tokens: u32_of(u, "input_tokens") + cache_r + cache_w,
                        output_tokens: u32_of(u, "output_tokens"),
                        cache_read_tokens: cache_r,
                        cache_write_tokens: cache_w,
                        reasoning_tokens: 0,
                    }
                });
                out.push(CoreStreamEvent::MessageDelta {
                    stop_reason: sr,
                    usage,
                });
            }
            "message_stop" => out.push(CoreStreamEvent::MessageStop),
            "error" => {
                let message = data
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("upstream stream error")
                    .to_string();
                out.push(CoreStreamEvent::Error { message });
            }
            // ping 及未知事件忽略
            _ => {}
        }

        Ok(out)
    }
}

fn client_sse(event: &str, data: Value) -> RawChunk {
    RawChunk::json(
        ChunkStage::ClientChunk,
        Protocol::Anthropic,
        Some(event.to_string()),
        data,
    )
}

#[async_trait]
impl ClientStreamAdapter for AnthropicAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::Anthropic
    }

    /// Core 流事件 → Anthropic Messages SSE（与上游 decode 互为逆向）。
    ///
    /// 推理块惰性开启：上游 reasoning item 可能零明文零凭据（如仅含
    /// encrypted_content 且凭据增量未到时不可预知）。若在 BlockStart 直接
    /// 开 thinking 块，零 delta 收尾会产生空 thinking 块（违反 Anthropic
    /// 规范，客户端报 reasoning part not found）。因此：
    ///   * 第一个 thinking_delta 到达时才开 thinking 块；
    ///   * 凭据先于明文到达时，按 redacted_thinking 完整块形态开启；
    ///   * 全程无内容的推理块不产生任何事件。
    fn encode(
        &self,
        _ctx: &ReqCtx,
        ev: &CoreStreamEvent,
        st: &mut StreamEncodeState,
    ) -> Result<Vec<RawChunk>> {
        let mut out = Vec::new();
        match ev {
            CoreStreamEvent::MessageStart { id, model } => {
                out.push(client_sse(
                    "message_start",
                    json!({
                        "type": "message_start",
                        "message": {
                            "id": id, "type": "message", "role": "assistant", "model": model,
                            "content": [], "stop_reason": null, "stop_sequence": null,
                            "usage": { "input_tokens": 0, "output_tokens": 0 },
                        },
                    }),
                ));
            }
            CoreStreamEvent::BlockStart { index, block } => {
                // 推理块不开：等首个 delta 到达再惰性开启（见 fn 文档）；
                // 仍记下起始块——内容只在收尾块到达时 BlockStop 兜底整段下发。
                if matches!(block, ContentBlock::Reasoning { .. }) {
                    st.note_start(*index, block);
                    return Ok(out);
                }
                let cb = match block {
                    ContentBlock::ToolUse { id, name, .. } => {
                        // 记下起始块：收尾时若 block=None 仍须识别 tool_use
                        // 并补发完整 arguments（见 BlockStop 分支）。
                        st.note_start(*index, block);
                        json!({ "type": "tool_use", "id": id, "name": name, "input": {} })
                    }
                    _ => json!({ "type": "text", "text": "" }),
                };
                st.open_blocks.insert(*index);
                out.push(client_sse(
                    "content_block_start",
                    json!({ "type": "content_block_start", "index": index, "content_block": cb }),
                ));
            }
            CoreStreamEvent::BlockDelta { index, delta } => {
                match delta {
                    StreamDelta::Text { text } => {
                        st.push_delta(*index, text);
                        // 惰性开块：chat 系上游（OpenAI Chat 等）正文只有裸
                        // BlockDelta、无 BlockStart，必须补 content_block_start，
                        // 否则客户端收到没有块头的孤立增量。上游已发过
                        // BlockStart（anthropic/gemini/responses 路径）时
                        // open_blocks 已含该 index，insert 返回 false 不重复。
                        if st.open_blocks.insert(*index) {
                            out.push(client_sse(
                                "content_block_start",
                                json!({
                                    "type": "content_block_start", "index": index,
                                    "content_block": { "type": "text", "text": "" }
                                }),
                            ));
                        }
                        out.push(client_sse(
                            "content_block_delta",
                            json!({
                                "type": "content_block_delta", "index": index,
                                "delta": { "type": "text_delta", "text": text }
                            }),
                        ));
                    }
                    StreamDelta::ToolInput { partial_json } => {
                        st.push_delta(*index, partial_json);
                        out.push(client_sse(
                            "content_block_delta",
                            json!({
                                "type": "content_block_delta", "index": index,
                                "delta": { "type": "input_json_delta", "partial_json": partial_json }
                            }),
                        ));
                    }
                    StreamDelta::Reasoning { text } => {
                        st.push_delta(*index, text);
                        // 惰性开启：首个推理明文增量到达时才开 thinking 块
                        if st.open_blocks.insert(*index) {
                            out.push(client_sse(
                                "content_block_start",
                                json!({
                                    "type": "content_block_start", "index": index,
                                    "content_block": { "type": "thinking", "thinking": "" }
                                }),
                            ));
                        }
                        out.push(client_sse(
                            "content_block_delta",
                            json!({
                                "type": "content_block_delta", "index": index,
                                "delta": { "type": "thinking_delta", "thinking": text }
                            }),
                        ));
                    }
                    // 加密 CoT 回传凭据：客户端在 content_block_stop 前累积进
                    // thinking 块的 signature，下一轮原样回传。若凭据先于任何
                    // 明文到达，该块按 redacted_thinking 完整块形态开启。
                    // 出站解除标记：本家凭据还原原文；异源凭据（gem:/oai:）带标记
                    // 原样透传——回传入站时幂等打标记、回到归属协议上游才解除标记。
                    StreamDelta::ReasoningSignature { signature } => {
                        let sig = crate::adapters::emit_signature(
                            crate::adapters::SIG_ANTHROPIC,
                            signature.as_str(),
                        );
                        if st.open_blocks.insert(*index) {
                            out.push(client_sse(
                                "content_block_start",
                                json!({
                                    "type": "content_block_start", "index": index,
                                    "content_block": { "type": "redacted_thinking", "data": sig }
                                }),
                            ));
                        } else {
                            out.push(client_sse(
                                "content_block_delta",
                                json!({
                                    "type": "content_block_delta", "index": index,
                                    "delta": { "type": "signature_delta", "signature": sig }
                                }),
                            ));
                        }
                    }
                }
            }
            CoreStreamEvent::BlockStop { index, block } => {
                // 仅对已开启的块收尾；从未开启的（空推理块）不产生任何事件
                if st.open_blocks.remove(index) {
                    // 工具块的 arguments 可能只在收尾完整块里到达（Responses
                    // `arguments.done` 单帧形态）或增量帧全为空串：若本块从未
                    // 下发过非空 input_json_delta，先补一帧携带完整 input 的
                    // delta，再发 stop——否则客户端收到「start+stop、参数缓冲
                    // 为空」的畸形 tool_use，严格解析器直接判 InvalidStreamError
                    // 断开连接（qwen-code 实测行为）。Text 同理：上游把整段
                    // 正文只放在收尾块时补发 text_delta 再收尾。
                    let input = match block.as_ref() {
                        Some(ContentBlock::ToolUse { input, .. }) => Some(input),
                        _ => None,
                    };
                    let is_tool = input.is_some()
                        || matches!(st.blocks.get(index), Some(ContentBlock::ToolUse { .. }));
                    let pending_text = match block.as_ref() {
                        Some(ContentBlock::Text { text }) if !text.is_empty() => Some(text.clone()),
                        _ => None,
                    };
                    if let Some(text) = pending_text
                        .filter(|_| st.delta_acc.get(index).is_none_or(|s| s.is_empty()))
                    {
                        out.push(client_sse(
                            "content_block_delta",
                            json!({
                                "type": "content_block_delta", "index": index,
                                "delta": { "type": "text_delta", "text": text }
                            }),
                        ));
                    }
                    if is_tool {
                        let emitted = st.delta_acc.get(index).is_some_and(|s| !s.is_empty());
                        if !emitted {
                            let args = input
                                .filter(|v| !v.is_null())
                                .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "{}".into()))
                                .unwrap_or_else(|| "{}".to_string());
                            out.push(client_sse(
                                "content_block_delta",
                                json!({
                                    "type": "content_block_delta", "index": index,
                                    "delta": { "type": "input_json_delta", "partial_json": args }
                                }),
                            ));
                        }
                    }
                    out.push(client_sse(
                        "content_block_stop",
                        json!({ "type": "content_block_stop", "index": index }),
                    ));
                } else {
                    // 推理块惰性开启、内容只在收尾块里到达（上游不走明文增量）
                    // 时从未开过——把完整思考连开带收一次下发，否则静默丢失。
                    let thinking = match block.as_ref().or_else(|| st.blocks.get(index)) {
                        Some(ContentBlock::Reasoning {
                            text,
                            signature,
                            redacted,
                        }) => Some((text.clone(), signature.clone(), *redacted)),
                        _ => None,
                    };
                    if let Some((text, signature, redacted)) = thinking {
                        // 纯空块（无文本无凭据）不产出任何事件
                        if text.is_empty() && signature.is_none() {
                            return Ok(out);
                        }
                        if redacted {
                            let data = signature
                                .as_deref()
                                .map(|s| {
                                    crate::adapters::emit_signature(
                                        crate::adapters::SIG_ANTHROPIC,
                                        s,
                                    )
                                })
                                .unwrap_or_default();
                            out.push(client_sse(
                                "content_block_start",
                                json!({
                                    "type": "content_block_start", "index": index,
                                    "content_block": { "type": "redacted_thinking", "data": data }
                                }),
                            ));
                        } else {
                            out.push(client_sse(
                                "content_block_start",
                                json!({
                                    "type": "content_block_start", "index": index,
                                    "content_block": { "type": "thinking", "thinking": "" }
                                }),
                            ));
                            if !text.is_empty() {
                                out.push(client_sse(
                                    "content_block_delta",
                                    json!({
                                        "type": "content_block_delta", "index": index,
                                        "delta": { "type": "thinking_delta", "thinking": text }
                                    }),
                                ));
                            }
                            if let Some(sig) = signature.as_deref() {
                                let enc = crate::adapters::emit_signature(
                                    crate::adapters::SIG_ANTHROPIC,
                                    sig,
                                );
                                if !enc.is_empty() {
                                    out.push(client_sse(
                                        "content_block_delta",
                                        json!({
                                            "type": "content_block_delta", "index": index,
                                            "delta": { "type": "signature_delta", "signature": enc }
                                        }),
                                    ));
                                }
                            }
                        }
                        out.push(client_sse(
                            "content_block_stop",
                            json!({ "type": "content_block_stop", "index": index }),
                        ));
                    }
                }
            }
            CoreStreamEvent::MessageDelta { stop_reason, usage } => {
                if let Some(u) = usage {
                    st.merge_usage(u);
                }
                let mut data = json!({
                    "type": "message_delta",
                    // stop_reason 未知（上游 message_start 的早段 usage 更新）时保持
                    // null——伪造 "end_turn" 会让客户端在流中段误判回合结束。
                    "delta": {
                        "stop_reason": stop_reason.map(unmap_stop_reason),
                        "stop_sequence": null
                    },
                });
                // 官方 message_delta.usage 为累计口径：发跨 delta 合并后的累计值——
                // 上游（OpenAI 系）仅在流末给出用量，Anthropic 上游则把 input 放在
                // message_start；客户端（Claude Code 等）靠它管理上下文，缺了
                // input_tokens 会退化为无法估算上下文窗口占用。
                // usage 字段本身必填（Anthropic SDK 模型要求），上游未上报时发零值。
                data["usage"] = anthropic_usage_out(&st.usage_acc);
                out.push(client_sse("message_delta", data));
            }
            CoreStreamEvent::MessageStop => {
                out.push(client_sse(
                    "message_stop",
                    json!({ "type": "message_stop" }),
                ));
            }
            CoreStreamEvent::Error { message } => {
                out.push(client_sse(
                    "error",
                    json!({ "type": "error", "error": { "type": "api_error", "message": message } }),
                ));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 上游 signature_delta → Core 凭据增量；入口 encode 还原 signature_delta。
    #[test]
    fn signature_delta_roundtrips() {
        let adapter = AnthropicAdapter;
        let ctx = ReqCtx::new("r1", Protocol::Anthropic);
        let evs = adapter
            .decode(
                &ctx,
                &mut StreamDecodeState::default(),
                &chunk(json!({
                    "type": "content_block_delta", "index": 0,
                    "delta": { "type": "signature_delta", "signature": "SIG" }
                })),
            )
            .unwrap();
        match &evs[0] {
            CoreStreamEvent::BlockDelta {
                delta: StreamDelta::ReasoningSignature { signature },
                ..
            } => {
                assert_eq!(signature, "ant:SIG", "Core 内部凭据带来源标记");
            }
            other => panic!("expected signature delta, got {other:?}"),
        }
        let mut c = ReqCtx::new("r1", Protocol::Anthropic);
        c.model_alias = "m".into();
        // 凭据先于任何明文到达：按 redacted_thinking 完整块形态开启（惰性开块）
        let chunks = adapter
            .encode(
                &c,
                &CoreStreamEvent::BlockDelta {
                    index: 0,
                    delta: StreamDelta::ReasoningSignature {
                        signature: "SIG".into(),
                    },
                },
                &mut StreamEncodeState::default(),
            )
            .unwrap();
        assert_eq!(chunks.len(), 1);
        let cb = &chunks[0].data.as_json().unwrap();
        assert_eq!(cb["type"], "content_block_start");
        assert_eq!(cb["content_block"]["type"], "redacted_thinking");
        assert_eq!(cb["content_block"]["data"], "SIG");

        // 块已开启（明文已流出）：凭据增量正常发 signature_delta
        let mut st = StreamEncodeState::default();
        adapter
            .encode(
                &c,
                &CoreStreamEvent::BlockDelta {
                    index: 0,
                    delta: StreamDelta::Reasoning {
                        text: "think".into(),
                    },
                },
                &mut st,
            )
            .unwrap();
        let chunks = adapter
            .encode(
                &c,
                &CoreStreamEvent::BlockDelta {
                    index: 0,
                    delta: StreamDelta::ReasoningSignature {
                        signature: "SIG".into(),
                    },
                },
                &mut st,
            )
            .unwrap();
        assert_eq!(chunks.len(), 1);
        let d = &chunks[0].data.as_json().unwrap();
        assert_eq!(d["type"], "content_block_delta");
        assert_eq!(d["delta"]["type"], "signature_delta");
        assert_eq!(d["delta"]["signature"], "SIG");
    }

    /// 空推理块（零明文零凭据）：不产生任何 content_block 事件，
    /// 避免空 thinking 块（客户端报 reasoning part not found）。
    #[test]
    fn empty_reasoning_block_emits_nothing() {
        let adapter = AnthropicAdapter;
        let mut ctx = ReqCtx::new("r1", Protocol::Anthropic);
        ctx.model_alias = "m".into();
        let mut st = StreamEncodeState::default();

        let chunks = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStart {
                    index: 0,
                    block: ContentBlock::Reasoning {
                        text: String::new(),
                        signature: None,
                        redacted: false,
                    },
                },
                &mut st,
            )
            .unwrap();
        assert!(chunks.is_empty(), "BlockStart 不应开块");

        let chunks = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStop {
                    index: 0,
                    block: None,
                },
                &mut st,
            )
            .unwrap();
        assert!(chunks.is_empty(), "未开启的块不应收尾");
    }

    fn chunk(v: Value) -> RawChunk {
        RawChunk::json(ChunkStage::UpstreamChunk, Protocol::Anthropic, None, v)
    }

    #[test]
    fn decodes_text_stream() {
        let adapter = AnthropicAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);
        let mut st = StreamDecodeState::default();

        let evs = adapter
            .decode(&ctx, &mut st, &chunk(json!({"type":"message_start","message":{"id":"m1","model":"claude","usage":{"input_tokens":7}}})))
            .unwrap();
        assert!(matches!(evs[0], CoreStreamEvent::MessageStart { .. }));

        let evs = adapter
            .decode(&ctx, &mut st, &chunk(json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})))
            .unwrap();
        assert!(matches!(
            evs[0],
            CoreStreamEvent::BlockStart { index: 0, .. }
        ));

        let evs = adapter
            .decode(&ctx, &mut st, &chunk(json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}})))
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

        // message_delta 携带完整 usage（server tool 等会让 input/cache 轮内增长）：
        // 全字段须被解析，聚合端 max 归并后新增量不丢。
        let evs = adapter
            .decode(&ctx, &mut st, &chunk(json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":9,"output_tokens":3,"cache_read_input_tokens":4,"cache_creation_input_tokens":2}})))
            .unwrap();
        match &evs[0] {
            CoreStreamEvent::MessageDelta { stop_reason, usage } => {
                assert_eq!(*stop_reason, Some(StopReason::EndTurn));
                let u = usage.unwrap();
                assert_eq!(u.output_tokens, 3);
                assert_eq!(u.input_tokens, 15, "input 须并入 cache（Core 口径总量）");
                assert_eq!(u.cache_read_tokens, 4);
                assert_eq!(u.cache_write_tokens, 2);
            }
            _ => panic!("expected message delta"),
        }
    }

    #[test]
    fn decodes_tool_use_delta() {
        let adapter = AnthropicAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);
        let evs = adapter
            .decode(&ctx, &mut StreamDecodeState::default(), &chunk(json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}})))
            .unwrap();
        match &evs[0] {
            CoreStreamEvent::BlockDelta {
                index,
                delta: StreamDelta::ToolInput { partial_json },
            } => {
                assert_eq!(*index, 1);
                assert_eq!(partial_json, "{\"a\":");
            }
            _ => panic!("expected tool input delta"),
        }
    }

    #[test]
    fn ignores_ping() {
        let adapter = AnthropicAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);
        let evs = adapter
            .decode(
                &ctx,
                &mut StreamDecodeState::default(),
                &chunk(json!({"type":"ping"})),
            )
            .unwrap();
        assert!(evs.is_empty());
    }

    /// 回归（qwen-code InvalidStreamError）：工具块的 arguments 只在收尾完整块
    /// 里到达时，encode 必须在 content_block_stop 前补发携带完整 input 的
    /// input_json_delta——否则下游收到「start+stop、参数缓冲为空」的畸形
    /// tool_use，严格客户端直接断连（实测 error_type=InvalidStreamError）。
    #[test]
    fn encodes_stop_with_args_when_no_input_delta() {
        use moonbridge_core::ContentBlock;
        let adapter = AnthropicAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);
        let mut st = StreamEncodeState::default();
        let block = ContentBlock::ToolUse {
            id: "call_1".into(),
            item_id: None,
            name: "list_directory".into(),
            namespace: None,
            input: json!({"path": "/tmp"}),
            signature: None,
        };
        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStart {
                    index: 0,
                    block: block.clone(),
                },
                &mut st,
            )
            .unwrap();
        assert_eq!(evs.len(), 1, "BlockStart 应只产 content_block_start");

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
        assert_eq!(
            evs.len(),
            2,
            "缺参数增量的 tool_use 收尾应补 delta + stop: {evs:?}"
        );
        let d = evs[0].data.as_json().unwrap();
        assert_eq!(d["type"], "content_block_delta");
        assert_eq!(d["delta"]["type"], "input_json_delta");
        assert_eq!(d["delta"]["partial_json"], r#"{"path":"/tmp"}"#);
        assert_eq!(evs[1].data.as_json().unwrap()["type"], "content_block_stop");
    }

    /// 同上但上游真·空参（无 block / input 为 null）：也要补 "{}" 而非发空缓冲。
    #[test]
    fn encodes_stop_with_empty_args_fallback() {
        use moonbridge_core::ContentBlock;
        let adapter = AnthropicAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);
        let mut st = StreamEncodeState::default();
        let block = ContentBlock::ToolUse {
            id: "call_2".into(),
            item_id: None,
            name: "get_time".into(),
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
        assert_eq!(
            evs.len(),
            2,
            "block=None 的 tool_use 收尾也应补 delta: {evs:?}"
        );
        assert_eq!(
            evs[0].data.as_json().unwrap()["delta"]["partial_json"],
            "{}"
        );
    }

    /// 已有非空参数增量时不应重复补发。
    #[test]
    fn no_duplicate_args_when_delta_streamed() {
        use moonbridge_core::ContentBlock;
        let adapter = AnthropicAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiResponse);
        let mut st = StreamEncodeState::default();
        let block = ContentBlock::ToolUse {
            id: "call_3".into(),
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
        assert_eq!(evs.len(), 1, "已下发参数增量的块收尾只应有 stop: {evs:?}");
        assert_eq!(evs[0].data.as_json().unwrap()["type"], "content_block_stop");
    }
}
