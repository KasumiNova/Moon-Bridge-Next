//! 内置协议 Adapter（四象限：入口 Client / 上游 Provider × 非流式 / 流式）。
//!
//! - [`openai_responses`]：OpenAI Responses（`/v1/responses`，入口 + 上游）。
//! - [`anthropic`]：Anthropic Messages（`/v1/messages`，入口 + 上游）。
//! - [`openai_chat`]：OpenAI Chat Completions（`/v1/chat/completions`，入口 + 上游）。
//! - [`google_genai`]：Google Generative AI / Gemini（入口 + 上游）。
//!
//! 本模块另放各家上游共用的 reasoning 映射辅助（见下）。

pub mod anthropic;
pub mod google_genai;
pub mod openai_chat;
pub mod openai_responses;

use moonbridge_core::CoreRequest;

/// Anthropic 扩展思考的最小预算（低于此值上游会拒绝请求）。
const MIN_THINKING_BUDGET: u32 = 1024;

/// effort → token 预算档位。
///
/// OpenAI 系用字符串枚举表达推理强度，Anthropic（`thinking.budget_tokens`）用
/// **token 预算**表达，Gemini 现行接口为 `thinkingLevel` 档位（历史 budget 形态
/// 仅在入站解析时经本表逆推 effort）。共用一张映射表翻译同一意图，避免四个
/// adapter 各写一份而漂移。
const THINKING_BUDGETS: &[(&str, u32)] = &[
    ("minimal", 1024),
    ("low", 2048),
    ("medium", 8192),
    ("high", 16384),
    ("xhigh", 24576),
    ("max", 32768),
];

/// 取本次请求有效的 reasoning effort；未设置或全空白时返回 `None`。
pub(crate) fn reasoning_effort(req: &CoreRequest) -> Option<&str> {
    req.reasoning
        .as_ref()
        .and_then(|r| r.effort.as_deref())
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// token 预算 → effort 档位（`thinking_budget` 的逆映射，取不超过预算的最大档）。
/// 供以预算表达思考强度的入口（anthropic/gemini）解析客户端配置。
pub(crate) fn effort_from_budget(budget: u32) -> &'static str {
    THINKING_BUDGETS
        .iter()
        .rev()
        .find(|(_, v)| budget >= *v)
        .map(|(k, _)| *k)
        .unwrap_or("minimal")
}

/// effort 对应的 token 预算；未知 effort 回落到 `medium` 档（保守：不因此关掉思考）。
fn thinking_budget(effort: &str) -> u32 {
    THINKING_BUDGETS
        .iter()
        .find(|(k, _)| effort.eq_ignore_ascii_case(k))
        .map(|(_, v)| *v)
        .unwrap_or(8192)
}

/// 换算并收紧到严格小于 `max_tokens` 的思考预算；收不出合法值时返回 `None`
/// （宁可不发，也不构造出必被上游拒绝的请求）。
pub(crate) fn clamped_thinking_budget(effort: &str, max_tokens: u32) -> Option<u32> {
    let budget = thinking_budget(effort).min(max_tokens.saturating_sub(1));
    (budget >= MIN_THINKING_BUDGET).then_some(budget)
}

// ============================================================================
// 推理凭据的来源标记
// ============================================================================
//
// `ContentBlock::Reasoning.signature` / `ToolUse.signature` 承载的是**不透明
// 回传凭据**（Anthropic thinking signature / OpenAI encrypted_content /
// Gemini thoughtSignature）。跨协议转发时把 A 家的凭据填进 B 家的字段会被
// 上游拒绝（或污染推理状态），故入站时给凭据打上来源标记，出站时只放行
// 同来源凭据、异源凭据按「无凭据」降级处理。
//
// 无前缀的凭据（插件注入、旧会话遗留）视为未知来源，保持透传——历史行为。

pub(crate) const SIG_ANTHROPIC: &str = "ant:";
pub(crate) const SIG_OPENAI: &str = "oai:";
pub(crate) const SIG_GEMINI: &str = "gem:";
/// Chat 协议（OpenAI Chat Completions）上游没有独立推理凭据字段——
/// thinking 模式上游（DeepSeek/Kimi 系）要求把 `reasoning_content` 原文
/// 随历史回传，**推理明文本身即凭据**。decode 侧给无凭据推理打上此前缀
/// （payload 为推理原文），使 Responses 等客户端方向有可回传的不透明
/// 凭据（`encrypted_content`）；同时前缀机制让该自凭据在 OpenAI/Anthropic/
/// Gemini 上游方向按异源凭据降级，不会以假凭据污染真上游。
pub(crate) const SIG_CHAT: &str = "chat:";

/// 入站：给凭据打上来源标记（空凭据原样返回 `None`）。已带已知前缀的
/// 凭据视为先前打标记的回传，原样保留（幂等）——凭据经客户端转一圈后
/// 还会回到本 adapter，二次打标记会让 `untag_signature` 错位。
pub(crate) fn tag_signature(prefix: &str, sig: Option<String>) -> Option<String> {
    sig.filter(|s| !s.is_empty()).map(|s| {
        if [SIG_ANTHROPIC, SIG_OPENAI, SIG_GEMINI, SIG_CHAT]
            .iter()
            .any(|p| s.starts_with(p))
        {
            s
        } else {
            format!("{prefix}{s}")
        }
    })
}

/// 出站：取回属于 `prefix` 的凭据原文。异源凭据返回 `None`（按无凭据降级）；
/// 无前缀凭据视为未知来源透传。
pub(crate) fn untag_signature<'a>(prefix: &str, sig: Option<&'a str>) -> Option<&'a str> {
    let s = sig?;
    if s.is_empty() {
        return None;
    }
    for p in [SIG_ANTHROPIC, SIG_OPENAI, SIG_GEMINI, SIG_CHAT] {
        if let Some(rest) = s.strip_prefix(p) {
            return (p == prefix).then_some(rest);
        }
    }
    Some(s)
}

/// 客户端方向凭据透传：本家凭据（`prefix`）还原原文；异源凭据**带标记**
/// 原样下发——客户端把它当不透明串存入历史，回传入站时幂等打标记，
/// 最终回到归属协议上游才由 `untag_signature` 解除标记。丢弃异源凭据会让
/// 「Gemini 上游 → 非 Gemini 客户端 → 历史回传 → Gemini 上游」断链。
pub(crate) fn emit_signature<'a>(prefix: &str, sig: &'a str) -> &'a str {
    untag_signature(prefix, Some(sig)).unwrap_or(sig)
}

#[cfg(test)]
mod tests {
    use super::*;
    use moonbridge_core::Reasoning;

    fn req_with(effort: Option<&str>) -> CoreRequest {
        let mut r = CoreRequest::new("m");
        r.reasoning = effort.map(|e| Reasoning {
            effort: Some(e.to_string()),
            summary: None,
        });
        r
    }

    #[test]
    fn effort_is_trimmed_and_blanks_are_none() {
        assert_eq!(reasoning_effort(&req_with(Some("  high "))), Some("high"));
        assert_eq!(reasoning_effort(&req_with(Some("   "))), None);
        assert_eq!(reasoning_effort(&req_with(None)), None);
        assert_eq!(reasoning_effort(&CoreRequest::new("m")), None);
    }

    #[test]
    fn budget_table_and_clamp() {
        assert_eq!(thinking_budget("low"), 2048);
        assert_eq!(thinking_budget("HIGH"), 16384, "大小写不敏感");
        assert_eq!(thinking_budget("bogus"), 8192, "未知档回落 medium");
        // 正常：预算小于 max_tokens
        assert_eq!(clamped_thinking_budget("low", 4096), Some(2048));
        // 收紧到 max_tokens - 1
        assert_eq!(clamped_thinking_budget("max", 4000), Some(3999));
        // 收不出合法值则不发思考配置
        assert_eq!(clamped_thinking_budget("high", 1024), None);
        assert_eq!(clamped_thinking_budget("high", 0), None);
    }
}
