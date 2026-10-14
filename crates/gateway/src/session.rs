//! 会话水印：把 session id 以 marker 形式嵌进助手**推理（CoT）块**的明文首部，
//! 靠客户端把整段对话历史原样带回，从而在下一轮被识别回同一会话。
//!
//! 为什么需要它：Qwen Code 这类客户端在请求头与请求体里都不携带会话标识（其
//! `prompt_cache_key` 注入路径以 hostname 恰为 `api.openai.com` 为前提，经网关时不生效），
//! 于是 `ctx.session_id` 恒为 `None`，插件的 `mb.session` 与 trace 的会话目录都失去粒度。
//!
//! 关键设计：**入站即剥除，剥完才转发上游**。上游模型的输入里永远不出现 marker，因此
//! 模型没有模仿它的机会（模仿只会发生在 marker 进入上下文时），marker 的唯一存活期是
//! 客户端本地 transcript。这也意味着发往上游的 prompt 比客户端存档更干净。
//!
//! 为什么嵌推理块而非正文：
//! - thinking 回传是硬语义——thinking 模式上游缺 `reasoning_content`/thinking 块
//!   直接 400，客户端必然原样带回，与 `<mb-cot>` 凭据机制同一条可靠通道；
//! - 每个 thinking 响应都带推理块（含 tool_use 轮），首轮起即可打标记；
//! - 正文是用户可见输出且会被客户端包进 tool_result 等结构，不再被 marker 污染。
//!
//! marker 形态：`[mb:<载荷>]`。载荷即会话身份本身——
//! - 网关自分配会话：载荷 = session uuid，`ctx.session_id` 直接用载荷，
//!   活跃表淘汰 / 网关重启都不再造成 session id（及 `x-opencode-session` 等
//!   下游亲和头）漂移；
//! - 外部身份会话（body `session_id` / `previous_response_id` /
//!   `X-Codex-Window-Id`）：载荷 = [`tag_from_id`] 派生的定长 6-hex 短 tag，
//!   经 [`SessionTable`] 还原回外部 id。

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use moonbridge_core::{ContentBlock, CoreRequest, CoreResponse, CoreStreamEvent, StreamDelta};

const OPEN: &str = "[mb:";
const CLOSE: char = ']';
/// 短 tag 长度（6 位 hex = 24 bit；外部身份派生 tag 用，活跃表深度上限内碰撞概率可忽略）。
const TAG_LEN: usize = 6;

/// marker 本身（`[mb:<载荷>]`）。
pub fn marker_inline(payload: &str) -> String {
    format!("{OPEN}{payload}{CLOSE}")
}

/// 推理块首部嵌入形态：`" [mb:<载荷>]"`——前置空格与 marker 同属可剥片段，
/// 剥除后字节还原（见 [`strip_text`] 去除紧邻空格的逻辑）。
pub fn marker_head(payload: &str) -> String {
    format!(" {}", marker_inline(payload))
}

/// 短 tag 形态：定长 6 位小写 hex（外部身份经 `tag_from_id` 派生）。
fn is_short_tag(s: &str) -> bool {
    s.len() == TAG_LEN
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// uuid 形态：8-4-4-4-12 小写 hex+连字符（网关自分配 session id 的原样嵌入）。
fn is_uuid(s: &str) -> bool {
    s.len() == 36
        && s.bytes().enumerate().all(|(i, b)| match i {
            8 | 13 | 18 | 23 => b == b'-',
            _ => b.is_ascii_hexdigit() && !b.is_ascii_uppercase(),
        })
}

/// 合法 marker 载荷：uuid（自带身份）或 6-hex 短 tag（外部身份在表里的键）。
/// 其余形态一律不认——形似 bracket 的正文必须原样保留。
fn valid_payload(s: &str) -> bool {
    is_short_tag(s) || is_uuid(s)
}

/// 由任意 id 派生短 tag（FNV-1a 取低 `TAG_LEN` 个 hex 位，**定长**且稳定）。
///
/// 不能只挑 id 里现成的 hex 字符：外部 `session_id` 可能是任意字符串（例如
/// `"ext-sess-1"` 只剩 `e`/`e`/`1` 三个 hex），挑出来的 tag 长度不足，下一轮
/// 形态校验不认 ⇒ marker 剥不掉、逐轮在客户端 transcript 里累积。定长哈希
/// 保证「打得上，也剥得掉」，同时保持同一 id 稳定映射同一 tag。
fn tag_from_id(id: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in id.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    let mut s = format!("{:x}", h & ((1u64 << (TAG_LEN * 4)) - 1));
    if s.len() < TAG_LEN {
        s.insert_str(0, &"0".repeat(TAG_LEN - s.len()));
    }
    s
}

/// 从文本中剥除所有合法 marker，返回清洗后的文本与**按出现顺序**收集的载荷。
fn strip_text(s: &str) -> (String, Vec<String>) {
    let mut out = String::with_capacity(s.len());
    let mut tags = Vec::new();
    let mut rest = s;
    while let Some(pos) = rest.find(OPEN) {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + OPEN.len()..];
        match after.find(CLOSE) {
            Some(end) if valid_payload(&after[..end]) => {
                tags.push(after[..end].to_string());
                // 一并移除 marker 前紧邻的那个空格（若有），避免留下悬挂空白
                if out.ends_with(' ') {
                    out.pop();
                }
                rest = &after[end + CLOSE.len_utf8()..];
            }
            // 形似但不是我们的 marker（如正文里的 `[mb:abc]`）：原样保留，冒号后内容照抄
            _ => {
                out.push_str(OPEN);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    (out, tags)
}

/// 遍历内容块，剥除其中的 marker；被剥成空块的（原本只装了 marker）一并删掉。
///
/// 覆盖 `Text`、`Reasoning.text` 与 `Reasoning.signature`：
/// - 旧式正文尾部 marker 与新式 CoT 首部 marker 都在 `text` 上；
/// - `chat:` 自凭据的载荷就是推理明文，marker 可能一并混在 signature 里；
///   真凭据（ant:/oai:/gem:）是 opaque blob、字符集不含 `[`，剥除对它们恒为 no-op。
///
/// 删空块是必要的：marker 独占的幻影块剥完只剩空壳，Anthropic 会拒收空文本块；
/// 对 Reasoning 块只删「剥空且无凭据」的——带 signature 的空文本块是合法凭据载体。
/// 保底不删到 `content` 变空——那同样是上游非法输入，宁可留一个空块也不能交出空数组。
/// `redacted` 块不剥不删：它是不可解凭据原形态，动了上游必拒。
fn strip_blocks(blocks: &mut Vec<ContentBlock>, found: &mut Vec<String>) {
    let mut emptied = Vec::new();
    for (idx, b) in blocks.iter_mut().enumerate() {
        match b {
            ContentBlock::Text { text } => {
                let (cleaned, tags) = strip_text(text);
                if tags.is_empty() {
                    continue;
                }
                *text = cleaned;
                found.extend(tags);
                if text.is_empty() {
                    emptied.push(idx);
                }
            }
            ContentBlock::Reasoning {
                text,
                signature,
                redacted,
            } => {
                if *redacted {
                    continue;
                }
                let (cleaned, tags) = strip_text(text);
                if !tags.is_empty() {
                    *text = cleaned;
                    found.extend(tags);
                    if text.is_empty() && signature.is_none() {
                        emptied.push(idx);
                    }
                }
                if let Some(sig) = signature {
                    let (cleaned_sig, stags) = strip_text(sig);
                    if !stags.is_empty() {
                        *sig = cleaned_sig;
                        found.extend(stags);
                    }
                }
            }
            ContentBlock::ToolResult { content, .. } => strip_blocks(content, found),
            _ => {}
        }
    }
    for idx in emptied.into_iter().rev() {
        if blocks.len() > 1 {
            blocks.remove(idx);
        }
    }
}

/// 剥除请求内全部 marker，返回**最后一次**出现的载荷（= 最新一轮附加的那个）。
///
/// 取最后一个而非第一个：同一会话的 marker 恒定，历史里若混入陈旧载荷（会话被淘汰后
/// 重新分配）应以最近一次为准。
pub fn extract_from_request(req: &mut CoreRequest) -> Option<String> {
    let mut found = Vec::new();
    for m in &mut req.messages {
        strip_blocks(&mut m.content, &mut found);
    }
    strip_blocks(&mut req.system, &mut found);
    found.pop()
}

/// 把 marker 嵌进首个非 redacted 推理块的**明文首部**。返回是否嵌入成功。
///
/// 无可用推理块则不打标记（纯 CoT 方案）：不向正文插入无关文本，也不凭空造块——
/// 无 thinking 的轮次让会话身份顺延到下一个有推理块的响应。
pub fn tag_response(resp: &mut CoreResponse, payload: &str) -> bool {
    match resp.content.iter_mut().find_map(|b| match b {
        ContentBlock::Reasoning {
            text,
            redacted: false,
            ..
        } => Some(text),
        _ => None,
    }) {
        Some(text) => {
            text.insert_str(0, &marker_head(payload));
            true
        }
        None => false,
    }
}

/// 流式路径：向同 index 注入推理明文增量，使 marker 成为该 thinking 块的首部内容。
///
/// 紧随 `BlockStart`（或首个裸推理增量）下发：不新占块、不改块序、不拦截
/// `BlockStop`——客户端编码器按既有推理增量路径自然把它编进 thinking 块开头
/// （Anthropic 的惰性开块、`reasoning_content` 增量、responses 摘要增量同理）。
/// redacted/凭据先行的块不可注入：入口编码器会把它们开成 redacted_thinking
/// 完整块，再补明文增量会扰乱客户端块状态机。
pub fn marker_reasoning_delta(payload: &str, index: usize) -> CoreStreamEvent {
    CoreStreamEvent::BlockDelta {
        index,
        delta: StreamDelta::Reasoning {
            text: marker_head(payload),
        },
    }
}

/// 活跃会话表：marker 载荷 → session id，按命中刷新热度（LRU）淘汰，深度可配。
///
/// uuid 载荷自带身份（登记 `uuid→uuid` 仅为淘汰簿记）；短 tag 载荷须经表还原
/// 外部 id。淘汰即代表该会话「不再活跃」：被淘汰的 session id 反馈给调用方去
/// 清理插件侧的会话状态。
pub struct SessionTable {
    depth: usize,
    inner: Mutex<Inner>,
}

#[derive(Default)]
struct Inner {
    order: VecDeque<String>,
    by_key: HashMap<String, String>,
}

impl SessionTable {
    /// 以给定深度构造（`depth < 1` 时按 1 处理，避免表恒空导致会话永远无法复用）。
    pub fn new(depth: usize) -> Self {
        Self {
            depth: depth.max(1),
            inner: Mutex::new(Inner::default()),
        }
    }

    /// 用已有 session id 占位（供 `session_id` 字段 / `X-Codex-Window-Id` 等外部来源复用）。
    /// 返回 `(marker 载荷, 被淘汰的 session id 列表)`。
    pub fn note_external(&self, session_id: &str) -> (String, Vec<String>) {
        self.register(tag_from_id(session_id), session_id.to_string())
    }

    /// 新分配一个会话。返回 `(session_id, 被淘汰的 session id 列表, marker 载荷)`；
    /// 载荷即 session id 本身（uuid），入站回带时不查表即可还原身份。
    pub fn new_session(&self) -> (String, Vec<String>, String) {
        let id = uuid::Uuid::new_v4().to_string();
        let (_p, evicted) = self.register(id.clone(), id.clone());
        (id.clone(), evicted, id)
    }

    /// 解析客户端带回的 marker 载荷，返回 `(session id, 被淘汰的 session id 列表)`。
    ///
    /// - 命中表（外部身份绑定的短 tag，或已登记的 uuid）→ 返回映射的 id 并刷新热度；
    /// - 未命中 → 载荷就地成身份：uuid 原样采用、短 tag 合成 `mb-{tag}`——两种都
    ///   确定性稳定，淘汰/重启后带回同一 marker 仍落同一 session id。
    pub fn resolve(&self, payload: &str) -> (String, Vec<String>) {
        if let Some(id) = self.hit(payload) {
            return (id, Vec::new());
        }
        let id = if is_uuid(payload) {
            payload.to_string()
        } else {
            format!("mb-{payload}")
        };
        let (_p, evicted) = self.register(payload.to_string(), id.clone());
        (id, evicted)
    }

    /// 按载荷查活跃会话（测试与可观测性用；不刷新热度）。
    pub fn lookup(&self, key: &str) -> Option<String> {
        self.inner.lock().ok()?.by_key.get(key).cloned()
    }

    /// 当前活跃会话数（测试与可观测性用）。
    pub fn len(&self) -> usize {
        self.inner.lock().map(|i| i.order.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 命中查询 + 热度刷新：把载荷挪到队尾，活跃会话不被误挤。
    fn hit(&self, key: &str) -> Option<String> {
        let mut g = self.inner.lock().ok()?;
        let id = g.by_key.get(key)?.clone();
        g.order.retain(|t| t != key);
        g.order.push_back(key.to_string());
        Some(id)
    }

    /// 登记 `载荷 → id`，超深则按 LRU 淘汰最久未命中的，返回**实际生效的载荷** 与被淘汰的 session id。
    fn register(&self, key: String, id: String) -> (String, Vec<String>) {
        let Ok(mut g) = self.inner.lock() else {
            return (key, Vec::new());
        };
        if g.by_key.get(&key) == Some(&id) {
            // 幂等命中：挪到队尾刷新热度（LRU），活跃会话不被误挤
            g.order.retain(|t| t != &key);
            g.order.push_back(key.clone());
            return (key, Vec::new());
        }
        // 短 tag 冲突（不同 id 抢同一 tag）：重新生成一个，不要覆盖掉已有会话。
        // 先按 id+序号确定性重生成（可复现），连续冲突 32 次才退化到随机 uuid——表深远小于
        // tag 空间（16.7M），这条路实际走不到，只是保证不会静默覆盖别人的会话。
        let mut key = key;
        let mut tries = 0u32;
        while g.by_key.contains_key(&key) {
            tries += 1;
            key = if tries <= 32 {
                tag_from_id(&format!("{id}\u{1f}{tries}"))
            } else {
                tag_from_id(&uuid::Uuid::new_v4().to_string())
            };
        }
        g.by_key.insert(key.clone(), id);
        g.order.push_back(key.clone());
        let mut evicted = Vec::new();
        while g.order.len() > self.depth {
            if let Some(old) = g.order.pop_front() {
                if let Some(gone) = g.by_key.remove(&old) {
                    evicted.push(gone);
                }
            }
        }
        (key, evicted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moonbridge_core::{Message, Role};

    fn text_req(texts: &[&str]) -> CoreRequest {
        let mut req = CoreRequest::new("m");
        for t in texts {
            req.messages.push(Message::text(Role::Assistant, *t));
        }
        req
    }

    fn reasoning_req(thinking: &str, signature: Option<&str>) -> CoreRequest {
        let mut req = CoreRequest::new("m");
        req.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Reasoning {
                text: thinking.to_string(),
                signature: signature.map(str::to_string),
                redacted: false,
            }],
            ext: Default::default(),
        });
        req
    }

    const UUID: &str = "550e8400-e29b-41d4-a716-446655440000";

    #[test]
    fn strip_removes_markers_and_returns_payloads_in_order() {
        let mut req = text_req(&[&format!("hello {}", marker_inline("a1b2c3"))]);
        let tag = extract_from_request(&mut req);
        assert_eq!(tag.as_deref(), Some("a1b2c3"));
        let ContentBlock::Text { text } = &req.messages[0].content[0] else {
            panic!("应为文本块");
        };
        assert_eq!(text, "hello", "marker 与其前置空格都应剥除干净: {text:?}");
    }

    /// uuid 载荷（自带身份的 marker 形态）也要认得、剥得净。
    #[test]
    fn strip_removes_uuid_payload_markers() {
        let mut req = reasoning_req(&format!(" {}hmm", marker_inline(UUID)), None);
        assert_eq!(extract_from_request(&mut req).as_deref(), Some(UUID));
        let ContentBlock::Reasoning { text, .. } = &req.messages[0].content[0] else {
            panic!("应为推理块");
        };
        assert_eq!(text, "hmm", "CoT 首部 marker 剥除后应字节还原: {text:?}");
    }

    #[test]
    fn strip_keeps_last_payload_when_history_has_stale_ones() {
        let mut req = text_req(&[
            &format!("old {}", marker_inline("111111")),
            &format!("newer {}", marker_inline(UUID)),
        ]);
        assert_eq!(extract_from_request(&mut req).as_deref(), Some(UUID));
        let joined: String = req
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(!joined.contains("mb:"), "所有 marker 都该剥掉: {joined:?}");
    }

    /// 形似但非法的 bracket 文本必须原样保留，不能误伤用户/模型内容。
    #[test]
    fn lookalike_brackets_survive() {
        for case in [
            "[mb:abc]",
            "[mb:GGGGGG]",
            "[mb:1234567]",
            "[MB:123456]",
            "[mb:123456",
        ] {
            let mut req = text_req(&[case]);
            assert_eq!(
                extract_from_request(&mut req),
                None,
                "{case} 不该被认作 marker"
            );
            let ContentBlock::Text { text } = &req.messages[0].content[0] else {
                panic!()
            };
            assert_eq!(text, case, "{case} 应原样保留");
        }
    }

    #[test]
    fn marker_in_tool_result_is_also_stripped() {
        let mut req = CoreRequest::new("m");
        req.messages.push(Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "t1".into(),
                content: vec![ContentBlock::text(format!(
                    "out {}",
                    marker_inline("ffee00")
                ))],
                is_error: false,
            }],
            ext: Default::default(),
        });
        assert_eq!(extract_from_request(&mut req).as_deref(), Some("ffee00"));
    }

    /// `chat:` 自凭据的载荷是推理明文本身——marker 混进去也要剥，
    /// 否则凭据还原路径（text 空时取 payload）会把 marker 送上游。
    #[test]
    fn marker_inside_chat_self_credential_is_stripped() {
        let mut req = reasoning_req("hmm", Some(&format!("chat: {}", marker_inline(UUID))));
        assert_eq!(extract_from_request(&mut req).as_deref(), Some(UUID));
        let ContentBlock::Reasoning { signature, .. } = &req.messages[0].content[0] else {
            panic!()
        };
        assert_eq!(
            signature.as_deref(),
            Some("chat:"),
            "凭据内 marker 应剥除干净: {signature:?}"
        );
    }

    /// 真凭据（opaque blob，不含 `[`）剥除须为 no-op。
    #[test]
    fn real_signature_is_untouched_by_strip() {
        let mut req = reasoning_req("hmm", Some("ant:EqQBCgIYAh=="));
        assert_eq!(extract_from_request(&mut req), None);
        let ContentBlock::Reasoning { signature, .. } = &req.messages[0].content[0] else {
            panic!()
        };
        assert_eq!(signature.as_deref(), Some("ant:EqQBCgIYAh=="));
    }

    /// redacted 块是不可解凭据原形态：不剥不删（其 text 本就为空）。
    #[test]
    fn redacted_reasoning_is_never_touched() {
        let mut req = CoreRequest::new("m");
        req.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Reasoning {
                text: String::new(),
                signature: Some("ant:ENC".into()),
                redacted: true,
            }],
            ext: Default::default(),
        });
        assert_eq!(extract_from_request(&mut req), None);
        assert_eq!(req.messages[0].content.len(), 1, "redacted 块必须保留");
    }

    /// 只装了 marker 的幻影推理块（无凭据）剥完必须删掉——上游不认空 thinking。
    #[test]
    fn phantom_marker_only_reasoning_block_is_dropped() {
        let mut req = CoreRequest::new("m");
        req.messages.push(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Reasoning {
                    text: format!(" {}", marker_inline(UUID)),
                    signature: None,
                    redacted: false,
                },
                ContentBlock::text("Hello world"),
            ],
            ext: Default::default(),
        });
        assert_eq!(extract_from_request(&mut req).as_deref(), Some(UUID));
        let msg = &req.messages[0];
        assert_eq!(
            msg.content.len(),
            1,
            "空掉的水印块应被移除: {:?}",
            msg.content
        );
        let ContentBlock::Text { text } = &msg.content[0] else {
            panic!()
        };
        assert_eq!(text, "Hello world");
    }

    /// 只装了 marker 的文本块剥完必须删掉：流式旧版水印独占一个块，留下空文本块会被
    /// Anthropic 拒收（`text` 块不得为空）。
    #[test]
    fn marker_only_block_is_dropped_after_strip() {
        let mut req = CoreRequest::new("m");
        req.messages.push(Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::text("Hello world"),
                ContentBlock::text(marker_inline("abc123")),
            ],
            ext: Default::default(),
        });
        assert_eq!(extract_from_request(&mut req).as_deref(), Some("abc123"));
        let msg = &req.messages[0];
        assert_eq!(
            msg.content.len(),
            1,
            "空掉的水印块应被移除: {:?}",
            msg.content
        );
        let ContentBlock::Text { text } = &msg.content[0] else {
            panic!()
        };
        assert_eq!(text, "Hello world", "正文块不得受牵连");
    }

    /// 保底：消息只有一个块且它被剥空时也不得把 `content` 删成空数组（同样非法），
    /// 此时宁留一个空文本块。
    #[test]
    fn sole_marker_block_is_emptied_not_removed() {
        let mut req = CoreRequest::new("m");
        req.messages.push(Message {
            role: Role::Assistant,
            content: vec![ContentBlock::text(marker_inline("abc124"))],
            ext: Default::default(),
        });
        assert_eq!(extract_from_request(&mut req).as_deref(), Some("abc124"));
        assert_eq!(req.messages[0].content.len(), 1, "不得交出空 content");
    }

    #[test]
    fn tag_response_embeds_into_first_reasoning_block_head() {
        use moonbridge_core::CoreResponse;
        let mut resp = CoreResponse {
            id: "r".into(),
            model: "m".into(),
            content: vec![
                ContentBlock::Reasoning {
                    text: "hmm".into(),
                    signature: Some("ant:sig".into()),
                    redacted: false,
                },
                ContentBlock::text("sure"),
                ContentBlock::ToolUse {
                    id: "t".into(),
                    item_id: None,
                    name: "n".into(),
                    namespace: None,
                    input: serde_json::json!({}),
                    signature: None,
                },
            ],
            stop_reason: None,
            usage: Default::default(),
            ext: Default::default(),
        };
        assert!(
            tag_response(&mut resp, UUID),
            "tool_use 轮含推理块也应打标记"
        );
        let ContentBlock::Reasoning {
            text: think,
            signature,
            ..
        } = &resp.content[0]
        else {
            panic!()
        };
        assert_eq!(
            think,
            &format!("{}hmm", marker_head(UUID)),
            "marker 应在推理明文首部"
        );
        assert_eq!(signature.as_deref(), Some("ant:sig"), "凭据不得被改动");
        let ContentBlock::Text { text: body } = &resp.content[1] else {
            panic!()
        };
        assert_eq!(body, "sure", "正文块不得出现 marker");

        // 打上再剥必须字节还原
        let mut req = reasoning_req(think, signature.as_deref());
        assert_eq!(extract_from_request(&mut req).as_deref(), Some(UUID));
        let ContentBlock::Reasoning { text: back, .. } = &req.messages[0].content[0] else {
            panic!()
        };
        assert_eq!(back, "hmm", "打标记→剥除应还原原文");
    }

    /// 纯 CoT 方案：无可用推理块即不打标记——不向正文插入无关文本、不凭空造块。
    #[test]
    fn tag_response_skips_without_reasoning_and_redacted() {
        use moonbridge_core::CoreResponse;
        let mut no_reasoning = CoreResponse {
            id: "r".into(),
            model: "m".into(),
            content: vec![
                ContentBlock::text("sure"),
                ContentBlock::ToolUse {
                    id: "t".into(),
                    item_id: None,
                    name: "n".into(),
                    namespace: None,
                    input: serde_json::json!({}),
                    signature: None,
                },
            ],
            stop_reason: None,
            usage: Default::default(),
            ext: Default::default(),
        };
        assert!(!tag_response(&mut no_reasoning, UUID));
        assert_eq!(no_reasoning.content.len(), 2, "不得凭空造块");

        let mut only_redacted = CoreResponse {
            content: vec![ContentBlock::Reasoning {
                text: String::new(),
                signature: Some("ant:ENC".into()),
                redacted: true,
            }],
            ..no_reasoning.clone()
        };
        assert!(
            !tag_response(&mut only_redacted, UUID),
            "redacted 块不可注入"
        );
    }

    /// 流式 marker 是一条同 index 的推理明文增量——并入既有 thinking 块首部，
    /// 不占新块、无 start/stop 配对负担。
    #[test]
    fn marker_reasoning_delta_is_a_head_delta() {
        let ev = marker_reasoning_delta(UUID, 7);
        match &ev {
            CoreStreamEvent::BlockDelta { index, delta } => {
                assert_eq!(*index, 7);
                assert!(
                    matches!(delta, StreamDelta::Reasoning { text } if *text == marker_head(UUID)),
                    "增量应只含首部水印: {delta:?}"
                );
            }
            other => panic!("期望 BlockDelta，得到 {other:?}"),
        }
    }

    #[test]
    fn table_evicts_oldest_beyond_depth_and_reports_victims() {
        let t = SessionTable::new(2);
        let (a, ev_a, _a_p) = t.new_session();
        let (_b, ev_b, _b_p) = t.new_session();
        assert!(ev_a.is_empty() && ev_b.is_empty(), "未超深不应淘汰");
        let (_c, ev_c, _c_p) = t.new_session();
        assert_eq!(ev_c, vec![a], "LRU 应淘汰最久未命中的那个");
        assert_eq!(t.len(), 2, "表深被限制在 2");
    }

    #[test]
    fn hit_refreshes_position_so_active_session_survives() {
        let t = SessionTable::new(2);
        let (a, _e, a_p) = t.new_session();
        let (_b, _e, _b_p) = t.new_session();
        // 命中 a 刷新热度：新会话进来时应挤走 b 而非 a
        let (id, ev) = t.resolve(&a_p);
        assert_eq!(id, a, "uuid 载荷命中应还原同一 id");
        assert!(ev.is_empty());
        let (_c, ev_c, _c_p) = t.new_session();
        assert!(t.lookup(&a_p).is_some(), "刚命中过的会话不该被淘汰");
        assert_eq!(ev_c.len(), 1);
    }

    /// uuid 载荷不在表内（淘汰/重启）也不改身份：载荷即 session id。
    #[test]
    fn uuid_payload_is_self_identifying_across_eviction() {
        let t = SessionTable::new(1);
        let (a, _e, a_p) = t.new_session();
        let (_b, _e, _b_p) = t.new_session();
        assert!(t.lookup(&a_p).is_none(), "a 已被淘汰");
        let (id, _ev) = t.resolve(&a_p);
        assert_eq!(id, a, "淘汰后带回同一 uuid 仍须还原同一 session id");
    }

    /// 陈旧短 tag（外部绑定已丢失/重启）合成确定性身份并重绑：
    /// 同一 tag 反复回带恒落同一 `mb-{tag}`。
    #[test]
    fn stale_short_tag_derives_stable_identity() {
        let t = SessionTable::new(4);
        let (id1, _e) = t.resolve("abcdef");
        assert_eq!(id1, "mb-abcdef");
        let (id2, _e) = t.resolve("abcdef");
        assert_eq!(id2, "mb-abcdef", "重绑后命中应恒同");
    }

    #[test]
    fn lookup_hit_does_not_allocate_or_evict() {
        let t = SessionTable::new(3);
        let (id, _ev, p) = t.new_session();
        assert_eq!(t.lookup(&p).as_deref(), Some(id.as_str()));
        assert_eq!(t.len(), 1, "纯查询不得改变表大小");
    }

    #[test]
    fn tag_collision_does_not_hijack_existing_session() {
        let t = SessionTable::new(8);
        let (tag, _e) = t.note_external("ext-a");
        let first_id = t.lookup(&tag).expect("外部 id 应已登记");
        let (second_tag, _ev) = t.register(tag.clone(), uuid::Uuid::new_v4().to_string());
        assert_ne!(second_tag, tag, "不得覆盖已有 tag");
        assert_eq!(t.lookup(&tag).as_deref(), Some(first_id.as_str()));
    }

    #[test]
    fn depth_zero_is_clamped_to_one_not_dead_table() {
        let t = SessionTable::new(0);
        let (id, _ev, p) = t.new_session();
        assert_eq!(t.lookup(&p).as_deref(), Some(id.as_str()), "表不该恒空");
    }

    #[test]
    fn note_external_reuses_stable_tag_for_known_id() {
        let t = SessionTable::new(4);
        let id = "sess-abcdef123456";
        let (tag1, ev1) = t.note_external(id);
        let (tag2, ev2) = t.note_external(id);
        assert_eq!(tag1, tag2, "同一外部 id 应稳定映射到同一 tag");
        assert!(ev1.is_empty() && ev2.is_empty());
        assert_eq!(t.len(), 1);
    }

    /// 外部 `session_id` 未必是 uuid。旧实现只挑 id 里现成的 hex 字符，`"ext-sess-1"`
    /// 只剩 3 个 ⇒ 产出长度不足的 tag，下一轮形态校验不认、marker 剥不掉，
    /// 于是水印逐轮在客户端 transcript 里累积。定长哈希必须保证「打得上也剥得掉」。
    #[test]
    fn external_id_with_sparse_hex_still_round_trips() {
        let t = SessionTable::new(4);
        let (tag, _ev) = t.note_external("ext-sess-1");
        assert!(is_short_tag(&tag), "外部 id 派生的 tag 必须合法可剥: {tag}");

        let mut req = text_req(&[&format!("body {}", marker_inline(&tag))]);
        assert_eq!(
            extract_from_request(&mut req).as_deref(),
            Some(tag.as_str())
        );
        let ContentBlock::Text { text } = &req.messages[0].content[0] else {
            panic!()
        };
        assert_eq!(text, "body", "水印应被剥除干净、正文不受损");
        let (id, _e) = t.resolve(&tag);
        assert_eq!(id, "ext-sess-1", "短 tag 应还原外部 id");
    }

    /// 不同 id 应当分散到不同 tag（哈希退化时这条会先响）。
    /// 用确定性 id 样本：随机 uuid 采样在 24-bit 空间有约 0.8% 的碰撞率，
    /// 测试会偶发误报。
    #[test]
    fn tag_from_id_spreads_over_distinct_ids() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..512u32 {
            let id = format!("session-{i:08x}-fixed-sample");
            assert!(seen.insert(tag_from_id(&id)), "id {id} 的 tag 冲突异常频繁");
        }
        assert_eq!(seen.len(), 512);
    }
}
