//! Google Generative AI (Gemini) 流式：Core 事件 ↔ Gemini SSE chunk。
//!
//! Gemini `streamGenerateContent?alt=sse` 的每个 chunk 形如
//! `data: {"candidates":[{"content":{"parts":[{"text":"..."}],"role":"model"}}]}`，
//! 首块携带 `modelVersion`，末块携带 `finishReason` 与 `usageMetadata`；无 `[DONE]`。
//! decode 供 Gemini 作上游时解析，encode 供 Gemini 作入口时输出。
//!
//! Gemini 无 Anthropic 式的显式块边界事件，故 decode 以「首块」合成 MessageStart，
//! text/thought part 各自惰性开启独立块索引（thought → Reasoning 增量），
//! functionCall 经每流状态跨 chunk 单调分配索引，末块 finishReason 统一收尾。

use async_trait::async_trait;
use moonbridge_core::{ContentBlock, CoreStreamEvent, Protocol, Result, StreamDelta, Usage};
use serde_json::{json, Value};

use super::dto::{map_finish_reason, unmap_stop_reason, usage_from_gemini, usage_object};
use super::GoogleGenAiAdapter;
use crate::adapter::{
    ClientStreamAdapter, ProviderStreamAdapter, StreamDecodeState, StreamEncodeState,
};
use crate::context::ReqCtx;
use crate::raw::{ChunkStage, RawBody, RawChunk};

/// 从 chunk 取出 JSON data（[DONE] 等非 JSON 返回 None）。
fn chunk_json(chunk: &RawChunk) -> Option<Value> {
    if let Some(v) = chunk.data.as_json() {
        return Some(v.clone());
    }
    serde_json::from_str::<Value>(&chunk.raw).ok()
}

#[async_trait]
impl ProviderStreamAdapter for GoogleGenAiAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::GoogleGenai
    }

    fn decode(
        &self,
        _ctx: &ReqCtx,
        st: &mut StreamDecodeState,
        chunk: &RawChunk,
    ) -> Result<Vec<CoreStreamEvent>> {
        // 兼容网关可能补发 [DONE]
        if let RawBody::Text { text } = &chunk.data {
            if text.trim() == "[DONE]" {
                return Ok(vec![CoreStreamEvent::MessageStop]);
            }
        }
        let Some(data) = chunk_json(chunk) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();

        // 首块标记：Gemini 仅在首个 chunk 携带 modelVersion
        if data.get("modelVersion").is_some() && !st.message_started {
            st.message_started = true;
            let id = data
                .get("responseId")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let model = data
                .get("modelVersion")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            out.push(CoreStreamEvent::MessageStart { id, model });
            // 不再预开文本块：text/reasoning 块随首个对应 part 惰性开启，
            // 纯函数调用/纯思考响应不再留下空文本块。
        }

        // prompt 级拒绝/拦截：无 candidates、以 promptFeedback.blockReason
        // 终止。此前整块被丢弃 → 客户端收到空输出；现映射为内容过滤终止。
        if data.get("candidates").is_none()
            || data["candidates"]
                .as_array()
                .map(|a| a.is_empty())
                .unwrap_or(false)
        {
            if let Some(pf) = data.get("promptFeedback") {
                if let Some(br) = pf.get("blockReason").and_then(|v| v.as_str()) {
                    let usage = data.get("usageMetadata").map(usage_from_gemini);
                    out.push(CoreStreamEvent::MessageDelta {
                        stop_reason: Some(moonbridge_core::StopReason::ContentFilter),
                        usage,
                    });
                    out.push(CoreStreamEvent::Error {
                        message: format!("gemini prompt blocked: {br}"),
                    });
                    out.push(CoreStreamEvent::MessageStop);
                    return Ok(out);
                }
            }
            // candidates 缺失但携带 usageMetadata：中途用量上报（不伪造 stop_reason）
            if let Some(u) = data.get("usageMetadata") {
                out.push(CoreStreamEvent::MessageDelta {
                    stop_reason: None,
                    usage: Some(usage_from_gemini(u)),
                });
            }
            return Ok(out);
        }

        let candidate = data
            .get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
            .cloned()
            .unwrap_or(Value::Null);
        let parts = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default();

        for p in &parts {
            if let Some(fc) = p.get("functionCall") {
                let name = fc
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                // 无 id 时带块索引后缀合成——同名函数一次响应多次调用时
                // 纯 `{name}-call` 会重复，跨上游要求 tool_use id 唯一。
                let id = fc
                    .get("id")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .unwrap_or_else(|| format!("{name}-call-{}", st.next_block_index));
                let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                // 加密 CoT 凭据：与 function call 强绑定的 thoughtSignature
                // （part 级字段）。打上 gem: 来源标记，出站解除标记时异源凭据不互填。
                let signature = crate::adapters::tag_signature(
                    crate::adapters::SIG_GEMINI,
                    p.get("thoughtSignature")
                        .or_else(|| fc.get("thoughtSignature"))
                        .and_then(|v| v.as_str())
                        .map(String::from),
                );
                // 每个函数调用占独立块索引：Gemini 的 functionCall part 总是完整
                // 到达（args 不跨 chunk 切分），索引经 st.next_block_index 跨
                // chunk 单调分配，不与 text/reasoning 槽位或其他调用碰撞。
                // 非 Gemini 入口（Anthropic/Chat）没有 ToolUse 凭据位，
                // 故在调用前先发一个「仅凭据」载体推理块（即刻收尾）——
                // 客户端把它当推理凭据记入历史，回传时经 Core 还原到本 part。
                if let Some(sig) = &signature {
                    let c_idx = st.next_block_index;
                    st.next_block_index += 1;
                    let carrier = ContentBlock::Reasoning {
                        text: String::new(),
                        signature: Some(sig.clone()),
                        redacted: false,
                    };
                    out.push(CoreStreamEvent::BlockStart {
                        index: c_idx,
                        block: carrier.clone(),
                    });
                    out.push(CoreStreamEvent::BlockDelta {
                        index: c_idx,
                        delta: StreamDelta::ReasoningSignature {
                            signature: sig.clone(),
                        },
                    });
                    out.push(CoreStreamEvent::BlockStop {
                        index: c_idx,
                        block: Some(carrier),
                    });
                }
                let idx = st.next_block_index;
                st.next_block_index += 1;
                out.push(CoreStreamEvent::BlockStart {
                    index: idx,
                    block: ContentBlock::ToolUse {
                        id: id.clone(),
                        item_id: None,
                        name: name.clone(),
                        namespace: None,
                        input: json!({}),
                        signature: signature.clone(),
                    },
                });
                out.push(CoreStreamEvent::BlockDelta {
                    index: idx,
                    delta: StreamDelta::ToolInput {
                        partial_json: args.to_string(),
                    },
                });
                out.push(CoreStreamEvent::BlockStop {
                    index: idx,
                    block: Some(ContentBlock::ToolUse {
                        id,
                        item_id: None,
                        name,
                        namespace: None,
                        input: args,
                        signature,
                    }),
                });
                continue;
            }

            let thought = p.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
            let text = p.get("text").and_then(|v| v.as_str()).unwrap_or_default();
            // part 级 thoughtSignature：可能在带文本的 part 上，也可能在末块以
            // 空 text part 返回（官方文档：解析器必须检查空文本部分）。
            let sig = crate::adapters::tag_signature(
                crate::adapters::SIG_GEMINI,
                p.get("thoughtSignature")
                    .and_then(|v| v.as_str())
                    .map(String::from),
            );
            if text.is_empty() && sig.is_none() {
                continue;
            }
            // thought part → reasoning 槽位，普通 part → text 槽位，各自独占
            // 一个块索引（混入同一索引会让入口把思考明文写进 text 块）。
            // 空文本的仅凭据 part 不开任何槽位块——签名走载体。
            if !text.is_empty() {
                let (idx, newly) = st.slot(if thought { "reasoning" } else { "text" });
                if newly {
                    st.open_block(idx);
                    out.push(CoreStreamEvent::BlockStart {
                        index: idx,
                        block: if thought {
                            ContentBlock::Reasoning {
                                text: String::new(),
                                signature: None,
                                redacted: false,
                            }
                        } else {
                            ContentBlock::text("")
                        },
                    });
                }
                st.push_text(idx, thought, text);
                out.push(CoreStreamEvent::BlockDelta {
                    index: idx,
                    delta: if thought {
                        StreamDelta::Reasoning {
                            text: text.to_string(),
                        }
                    } else {
                        StreamDelta::Text {
                            text: text.to_string(),
                        }
                    },
                });
            }
            if let Some(sig) = sig {
                if thought {
                    // thought part 的签名是思考块自身属性：挂载到 reasoning 槽位
                    let (r_idx, r_new) = st.slot("reasoning");
                    if r_new {
                        st.open_block(r_idx);
                        out.push(CoreStreamEvent::BlockStart {
                            index: r_idx,
                            block: ContentBlock::Reasoning {
                                text: String::new(),
                                signature: None,
                                redacted: false,
                            },
                        });
                    }
                    st.push_signature(r_idx, &sig);
                    out.push(CoreStreamEvent::BlockDelta {
                        index: r_idx,
                        delta: StreamDelta::ReasoningSignature { signature: sig },
                    });
                } else {
                    // 普通 part 的签名（末块空 part 或文本 part 尾部）：
                    // 发「仅凭据」载体推理块（即刻收尾）。Core→Gemini 时按
                    // 「紧邻后继 ToolUse 优先、否则并入前一 part」规则还原。
                    let c_idx = st.next_block_index;
                    st.next_block_index += 1;
                    let carrier = ContentBlock::Reasoning {
                        text: String::new(),
                        signature: Some(sig.clone()),
                        redacted: false,
                    };
                    out.push(CoreStreamEvent::BlockStart {
                        index: c_idx,
                        block: carrier.clone(),
                    });
                    out.push(CoreStreamEvent::BlockDelta {
                        index: c_idx,
                        delta: StreamDelta::ReasoningSignature { signature: sig },
                    });
                    out.push(CoreStreamEvent::BlockStop {
                        index: c_idx,
                        block: Some(carrier),
                    });
                }
            }
        }

        // 末块：finishReason 触发所有已开块收尾 + MessageDelta(usage) + MessageStop
        if let Some(fr) = candidate.get("finishReason").and_then(|v| v.as_str()) {
            let mut open: Vec<usize> = st.open_blocks.clone();
            open.sort_unstable();
            st.open_blocks.clear();
            for idx in open {
                out.push(CoreStreamEvent::BlockStop {
                    index: idx,
                    block: st.blocks.get(&idx).cloned(),
                });
            }
            let usage = data.get("usageMetadata").map(usage_from_gemini);
            out.push(CoreStreamEvent::MessageDelta {
                stop_reason: map_finish_reason(fr),
                usage,
            });
            out.push(CoreStreamEvent::MessageStop);
        } else if let Some(u) = data.get("usageMetadata") {
            // 无 finishReason 的中途 usage chunk（Gemini 常在内容块间穿插上报）
            out.push(CoreStreamEvent::MessageDelta {
                stop_reason: None,
                usage: Some(usage_from_gemini(u)),
            });
        }

        Ok(out)
    }
}

fn gemini_chunk(parts: Value, finish: Option<&str>, usage: Option<Value>) -> RawChunk {
    let mut cand = json!({ "content": { "role": "model", "parts": parts }, "index": 0 });
    if let Some(fr) = finish {
        cand["finishReason"] = json!(fr);
    }
    let mut data = json!({ "candidates": [cand] });
    if let Some(u) = usage {
        data["usageMetadata"] = u;
    }
    RawChunk::json(ChunkStage::ClientChunk, Protocol::GoogleGenai, None, data)
}

#[async_trait]
impl ClientStreamAdapter for GoogleGenAiAdapter {
    fn protocol(&self) -> Protocol {
        Protocol::GoogleGenai
    }

    /// Core 流事件 → Gemini SSE。文本增量精确映射；Gemini 无起始/结束标记事件，
    /// 故 MessageStart/MessageStop 不产出 chunk，收尾信息随 MessageDelta 的
    /// finishReason + usageMetadata 下发。
    ///
    /// 局限：入口方向的工具参数增量（ToolInput）无状态难以累积为完整 `args`，
    /// 故 functionCall 在 BlockStart 时以已知 input 一次性下发（Gemini 入口为次要
    /// 链路，未挂载 HTTP 路由，主要用于矩阵完整性与测试）。
    fn encode(
        &self,
        _ctx: &ReqCtx,
        ev: &CoreStreamEvent,
        st: &mut StreamEncodeState,
    ) -> Result<Vec<RawChunk>> {
        let mut out = Vec::new();
        match ev {
            CoreStreamEvent::MessageStart { .. } => {}
            CoreStreamEvent::BlockStart { index, block } => {
                if let ContentBlock::ToolUse { .. } = block {
                    // 记下起始块：functionCall 的 args/thoughtSignature 可能只在
                    // 收尾完整块里到达（见 BlockStop 分支），此处暂不下发。
                    st.note_start(*index, block);
                }
            }
            CoreStreamEvent::BlockDelta { index, delta } => match delta {
                StreamDelta::Text { text } => {
                    if !text.is_empty() {
                        st.push_delta(*index, text);
                        out.push(gemini_chunk(json!([{ "text": text }]), None, None));
                    }
                }
                StreamDelta::Reasoning { text } => {
                    if !text.is_empty() {
                        st.push_delta(*index, text);
                        out.push(gemini_chunk(json!([{ "text": text }]), None, None));
                    }
                }
                // Gemini 流式无凭据承载位（thoughtSignature 是非流式 part 级字段）
                StreamDelta::ReasoningSignature { .. } => {}
                // 参数增量先累积——Gemini 的 functionCall args 不可跨 chunk 切分，
                // 只能随 BlockStop 一次性下发（见下）。
                StreamDelta::ToolInput { partial_json } => {
                    st.push_delta(*index, partial_json);
                }
            },
            CoreStreamEvent::BlockStop { index, block } => {
                // functionCall 是原子 part，等到收尾拿到完整 args 再一次性下发。
                // args 取法：收尾块 input > 累积增量拼合 > 起始块 input > 空对象——
                // 上游把参数只放在 arguments.done 类收尾帧时，起始块 input 是
                // 空壳，若在 BlockStart 就发会把真实参数整份丢掉。
                let tool = match block.as_ref() {
                    Some(ContentBlock::ToolUse { .. }) => block.as_ref(),
                    _ => st.blocks.get(index),
                };
                if let Some(ContentBlock::ToolUse { name, input, .. }) = tool {
                    let signature = match tool {
                        Some(ContentBlock::ToolUse {
                            signature: Some(s), ..
                        }) => Some(s.clone()),
                        _ => match st.blocks.get(index) {
                            Some(ContentBlock::ToolUse { signature, .. }) => signature.clone(),
                            _ => None,
                        },
                    };
                    let acc_args = st
                        .delta_acc
                        .get(index)
                        .filter(|s| !s.is_empty())
                        .and_then(|s| serde_json::from_str::<Value>(s).ok());
                    let start_input = match st.blocks.get(index) {
                        Some(ContentBlock::ToolUse { input: i, .. }) if !i.is_null() => {
                            Some(i.clone())
                        }
                        _ => None,
                    };
                    let args = if !input.is_null() {
                        input.clone()
                    } else {
                        acc_args.or(start_input).unwrap_or_else(|| json!({}))
                    };
                    let mut fc = json!({ "name": name, "args": args });
                    if let Some(sig) = crate::adapters::untag_signature(
                        crate::adapters::SIG_GEMINI,
                        signature.as_deref(),
                    ) {
                        if !sig.is_empty() {
                            fc["thoughtSignature"] = json!(sig);
                        }
                    }
                    out.push(gemini_chunk(json!([{ "functionCall": fc }]), None, None));
                    return Ok(out);
                }
                // Text/Reasoning 同理：整段只在收尾块到达时补发 text part。
                let text = match block.as_ref().or_else(|| st.blocks.get(index)) {
                    Some(ContentBlock::Text { text })
                    | Some(ContentBlock::Reasoning { text, .. }) => text.clone(),
                    _ => String::new(),
                };
                if !text.is_empty() && st.delta_acc.get(index).is_none_or(|s| s.is_empty()) {
                    out.push(gemini_chunk(json!([{ "text": text }]), None, None));
                }
            }
            CoreStreamEvent::MessageDelta { stop_reason, usage } => {
                if let Some(u) = usage {
                    st.merge_usage(u);
                }
                // stop_reason 未知（上游中途 usage 更新）时不携带 finishReason——
                // 伪造 STOP 会让客户端在流中段误判回合结束；usageMetadata 仍随
                // chunk 正常下发。
                let fr = stop_reason.map(|r| unmap_stop_reason(Some(r)));
                let usage = if st.usage_acc == Usage::default() {
                    None
                } else {
                    Some(usage_object(&st.usage_acc))
                };
                out.push(gemini_chunk(json!([]), fr, usage));
            }
            CoreStreamEvent::MessageStop => {}
            CoreStreamEvent::Error { message } => {
                out.push(RawChunk::json(
                    ChunkStage::ClientChunk,
                    Protocol::GoogleGenai,
                    None,
                    json!({ "error": { "message": message } }),
                ));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moonbridge_core::{StopReason, Usage};

    fn chunk(v: Value) -> RawChunk {
        RawChunk::json(ChunkStage::UpstreamChunk, Protocol::GoogleGenai, None, v)
    }

    /// 官方流式形态：签名可能在末块以空 text part 返回，需转为凭据增量，
    /// 挂载在 reasoning 槽位（未开启时先合成 BlockStart）。
    #[test]
    fn decodes_signature_only_trailing_part() {
        let adapter = GoogleGenAiAdapter;
        let ctx = ReqCtx::new("r1", Protocol::GoogleGenai);
        let mut st = StreamDecodeState::default();
        let evs = adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({
                    "candidates": [{ "content": { "role": "model", "parts": [
                        { "text": "", "thoughtSignature": "SIG" }
                    ] } }]
                })),
            )
            .unwrap();
        assert!(evs.iter().any(|e| matches!(
            e,
            CoreStreamEvent::BlockDelta { delta: StreamDelta::ReasoningSignature { signature }, .. } if signature == "gem:SIG"
        )), "凭据带来源标记");
    }

    #[test]
    fn decodes_first_text_chunk() {
        let adapter = GoogleGenAiAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        let mut st = StreamDecodeState::default();
        let evs = adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({
                    "candidates": [{ "content": { "role": "model", "parts": [{ "text": "Hello" }] }, "index": 0 }],
                    "modelVersion": "gemini-2.0-flash",
                    "responseId": "resp-1"
                })),
            )
            .unwrap();
        assert!(matches!(evs[0], CoreStreamEvent::MessageStart { .. }));
        assert!(matches!(
            evs[1],
            CoreStreamEvent::BlockStart { index: 0, .. }
        ));
        match &evs[2] {
            CoreStreamEvent::BlockDelta {
                index: 0,
                delta: StreamDelta::Text { text },
            } => {
                assert_eq!(text, "Hello")
            }
            other => panic!("expected text delta, got {other:?}"),
        }
    }

    #[test]
    fn middle_chunk_only_emits_delta() {
        let adapter = GoogleGenAiAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        // 首块已分配 text 槽位后，中间块只发增量、不重复合成 BlockStart
        let mut st = StreamDecodeState::default();
        adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({ "candidates": [{ "content": { "role": "model", "parts": [{ "text": "Hello" }] }, "index": 0 }] })),
            )
            .unwrap();
        let evs = adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({ "candidates": [{ "content": { "role": "model", "parts": [{ "text": " world" }] }, "index": 0 }] })),
            )
            .unwrap();
        assert_eq!(
            evs.len(),
            1,
            "中间块不应合成 MessageStart/BlockStart: {evs:?}"
        );
        assert!(matches!(
            evs[0],
            CoreStreamEvent::BlockDelta {
                index: 0,
                delta: StreamDelta::Text { .. }
            }
        ));
    }

    /// thought part 必须走独立 reasoning 块——混入 text 块会把 CoT 当正文下发。
    #[test]
    fn thought_parts_use_separate_reasoning_block() {
        let adapter = GoogleGenAiAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        let mut st = StreamDecodeState::default();
        let evs = adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({
                    "candidates": [{ "content": { "role": "model", "parts": [
                        { "thought": true, "text": "ponder" },
                        { "text": "answer" }
                    ] }, "index": 0 }]
                })),
            )
            .unwrap();
        let r_idx = evs
            .iter()
            .find_map(|e| match e {
                CoreStreamEvent::BlockDelta {
                    index,
                    delta: StreamDelta::Reasoning { text },
                } if text == "ponder" => Some(*index),
                _ => None,
            })
            .expect("thought part 应为 Reasoning 增量");
        let t_idx = evs
            .iter()
            .find_map(|e| match e {
                CoreStreamEvent::BlockDelta {
                    index,
                    delta: StreamDelta::Text { text },
                } if text == "answer" => Some(*index),
                _ => None,
            })
            .expect("普通 part 应为 Text 增量");
        assert_ne!(r_idx, t_idx, "reasoning 与 text 不得共用块索引");
    }

    /// 函数调用索引跨 chunk 单调分配：不与 text/reasoning 槽位碰撞。
    #[test]
    fn function_call_indexes_are_stable_across_chunks() {
        let adapter = GoogleGenAiAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        let mut st = StreamDecodeState::default();
        adapter
            .decode(
                &ctx,
                &mut st,
                &chunk(json!({ "candidates": [{ "content": { "role": "model", "parts": [{ "text": "a" }] }, "index": 0 }] })),
            )
            .unwrap();
        let mut idxs = Vec::new();
        for call in ["f1", "f2"] {
            let evs = adapter
                .decode(
                    &ctx,
                    &mut st,
                    &chunk(json!({
                        "candidates": [{ "content": { "role": "model", "parts": [{ "functionCall": { "name": call, "args": {} } }] }, "index": 0 }]
                    })),
                )
                .unwrap();
            let (idx, _) = evs
                .iter()
                .find_map(|e| match e {
                    CoreStreamEvent::BlockStart {
                        index,
                        block: ContentBlock::ToolUse { name, .. },
                    } => Some((*index, name.clone())),
                    _ => None,
                })
                .expect("应有 ToolUse BlockStart");
            idxs.push(idx);
        }
        assert_eq!(
            idxs,
            vec![1, 2],
            "跨 chunk 的函数调用须占递增且互不相同的块索引"
        );
    }

    #[test]
    fn decodes_finish_chunk() {
        let adapter = GoogleGenAiAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        let evs = adapter
            .decode(
                &ctx,
                &mut StreamDecodeState::default(),
                &chunk(json!({
                    "candidates": [{ "content": { "role": "model", "parts": [{ "text": "!" }] }, "finishReason": "STOP", "index": 0 }],
                    "usageMetadata": { "promptTokenCount": 8, "candidatesTokenCount": 4, "totalTokenCount": 12 }
                })),
            )
            .unwrap();
        // text delta, BlockStop(0), MessageDelta, MessageStop
        assert!(evs
            .iter()
            .any(|e| matches!(e, CoreStreamEvent::BlockDelta { .. })));
        assert!(evs
            .iter()
            .any(|e| matches!(e, CoreStreamEvent::BlockStop { index: 0, .. })));
        let md = evs.iter().find_map(|e| match e {
            CoreStreamEvent::MessageDelta { stop_reason, usage } => Some((stop_reason, usage)),
            _ => None,
        });
        let (sr, usage) = md.expect("应有 MessageDelta");
        assert_eq!(*sr, Some(StopReason::EndTurn));
        assert_eq!(
            usage.unwrap(),
            Usage {
                input_tokens: 8,
                output_tokens: 4,
                cache_read_tokens: 0,
                cache_write_tokens: 0,
                reasoning_tokens: 0
            }
        );
        assert!(matches!(evs.last().unwrap(), CoreStreamEvent::MessageStop));
    }

    #[test]
    fn decodes_function_call_chunk() {
        let adapter = GoogleGenAiAdapter;
        let ctx = ReqCtx::new("r1", Protocol::OpenAiChat);
        let evs = adapter
            .decode(
                &ctx,
                &mut StreamDecodeState::default(),
                &chunk(json!({
                    "candidates": [{ "content": { "role": "model", "parts": [{ "functionCall": { "name": "get_time", "args": {"tz":"UTC"} } }] }, "index": 0 }]
                })),
            )
            .unwrap();
        let bs = evs.iter().find_map(|e| match e {
            CoreStreamEvent::BlockStart {
                index,
                block: ContentBlock::ToolUse { name, .. },
            } => Some((*index, name.clone())),
            _ => None,
        });
        let (idx, name) = bs.expect("应有 ToolUse BlockStart");
        assert_eq!(idx, 0, "无前置文本块时函数调用从索引 0 起分配");
        assert_eq!(name, "get_time");
        assert!(evs.iter().any(|e| matches!(
            e,
            CoreStreamEvent::BlockDelta {
                delta: StreamDelta::ToolInput { .. },
                ..
            }
        )));
    }

    #[test]
    fn encodes_text_delta_and_finish() {
        let adapter = GoogleGenAiAdapter;
        let ctx = ReqCtx::new("r1", Protocol::GoogleGenai);

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
            evs[0].data.as_json().unwrap()["candidates"][0]["content"]["parts"][0]["text"],
            "Hi"
        );

        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::MessageDelta {
                    stop_reason: Some(StopReason::EndTurn),
                    usage: Some(Usage {
                        input_tokens: 3,
                        output_tokens: 2,
                        cache_read_tokens: 0,
                        cache_write_tokens: 0,
                        reasoning_tokens: 0,
                    }),
                },
                &mut StreamEncodeState::default(),
            )
            .unwrap();
        let data = evs[0].data.as_json().unwrap();
        assert_eq!(data["candidates"][0]["finishReason"], "STOP");
        assert_eq!(data["usageMetadata"]["promptTokenCount"], 3);
    }

    #[test]
    fn encode_message_start_and_stop_are_silent() {
        let adapter = GoogleGenAiAdapter;
        let ctx = ReqCtx::new("r1", Protocol::GoogleGenai);
        assert!(adapter
            .encode(
                &ctx,
                &CoreStreamEvent::MessageStart {
                    id: "x".into(),
                    model: "g".into()
                },
                &mut StreamEncodeState::default()
            )
            .unwrap()
            .is_empty());
        assert!(adapter
            .encode(
                &ctx,
                &CoreStreamEvent::MessageStop,
                &mut StreamEncodeState::default()
            )
            .unwrap()
            .is_empty());
    }

    /// 回归：functionCall 推迟到 BlockStop 一次性下发——上游把 args 只放在
    /// 收尾完整块里时（起始块 input 为空），必须以收尾块的完整 input 发出，
    /// 否则 functionCall 带着空 args 提前飞走、真实参数整份丢失。
    #[test]
    fn function_call_emitted_at_stop_with_complete_args() {
        use moonbridge_core::ContentBlock;
        let adapter = GoogleGenAiAdapter;
        let ctx = ReqCtx::new("r1", Protocol::GoogleGenai);
        let mut st = StreamEncodeState::default();
        let start_block = ContentBlock::ToolUse {
            id: "call_1".into(),
            item_id: None,
            name: "read_file".into(),
            namespace: None,
            input: Value::Null, // 上游起始块未带参数
            signature: None,
        };
        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStart {
                    index: 0,
                    block: start_block,
                },
                &mut st,
            )
            .unwrap();
        assert!(
            evs.is_empty(),
            "BlockStart 不应提前发 functionCall: {evs:?}"
        );

        let evs = adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStop {
                    index: 0,
                    block: Some(ContentBlock::ToolUse {
                        id: "call_1".into(),
                        item_id: None,
                        name: "read_file".into(),
                        namespace: None,
                        input: json!({"path": "/tmp/x"}),
                        signature: None,
                    }),
                },
                &mut st,
            )
            .unwrap();
        assert_eq!(evs.len(), 1);
        let fc =
            &evs[0].data.as_json().unwrap()["candidates"][0]["content"]["parts"][0]["functionCall"];
        assert_eq!(fc["name"], "read_file");
        assert_eq!(fc["args"]["path"], "/tmp/x");
    }

    /// 参数经增量流到达（block=None）时，收尾应从累积增量还原 args。
    #[test]
    fn function_call_args_from_delta_acc() {
        use moonbridge_core::ContentBlock;
        let adapter = GoogleGenAiAdapter;
        let ctx = ReqCtx::new("r1", Protocol::GoogleGenai);
        let mut st = StreamEncodeState::default();
        adapter
            .encode(
                &ctx,
                &CoreStreamEvent::BlockStart {
                    index: 0,
                    block: ContentBlock::ToolUse {
                        id: "call_2".into(),
                        item_id: None,
                        name: "write_file".into(),
                        namespace: None,
                        input: Value::Null,
                        signature: None,
                    },
                },
                &mut st,
            )
            .unwrap();
        for pj in [r#"{"path":"#, r#""/a""#, r#","c":"b"}"#] {
            adapter
                .encode(
                    &ctx,
                    &CoreStreamEvent::BlockDelta {
                        index: 0,
                        delta: StreamDelta::ToolInput {
                            partial_json: pj.into(),
                        },
                    },
                    &mut st,
                )
                .unwrap();
        }
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
        assert_eq!(evs.len(), 1);
        let fc =
            &evs[0].data.as_json().unwrap()["candidates"][0]["content"]["parts"][0]["functionCall"];
        assert_eq!(fc["args"]["path"], "/a");
        assert_eq!(fc["args"]["c"], "b");
    }
}
