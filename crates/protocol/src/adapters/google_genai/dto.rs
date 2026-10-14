//! Google Generative AI (Gemini) 的 DTO 与 Gemini ↔ Core 共享转换。
//!
//! 入口（client）与上游（provider）格式一致，故请求/响应的 contents、parts、
//! tools、usage、finishReason 映射集中于此，供四象限复用。
//!
//! Gemini 报文要点：
//! - 消息在 `contents`，每条 `{ role: "user"|"model", parts: [...] }`；
//! - system 走顶层 `systemInstruction`；
//! - part 种类：`text` / `inlineData` / `functionCall` / `functionResponse`；
//! - 工具声明在 `tools[].functionDeclarations`，选择策略在 `toolConfig.functionCallingConfig`；
//! - 采样参数集中在 `generationConfig`；用量在 `usageMetadata`。

use std::collections::HashMap;

use crate::adapters::reasoning_effort;
use moonbridge_core::{
    ContentBlock, CoreRequest, DocSource, Message, Role, StopReason, Tool, ToolChoice, Usage,
};
use serde_json::{json, Value};

/// 未显式指定协议版本时使用的 API 版本段。
pub const DEFAULT_VERSION: &str = "v1beta";

/// Gemini 服务端代码执行（`executableCode`/`codeExecutionResult` part）在
/// Core 侧的工具名承载：以 ToolUse/ToolResult 表达调用与结果，本常量作
/// 占位名，回传 Gemini 时还原为原生 part 形态。真实工具若恰好同名会
/// 误判，属可接受的边角冲突。
pub(crate) const EXEC_CODE: &str = "executable_code";

/// `functionCall` 无 `id` 字段时 ToolUse.id 的占位值；`functionResponse`
/// 无 `id` 时 ToolResult.tool_use_id 用 `{FR_NAME_PREFIX}{name}` 占位——
/// contents_to_core 收齐全部消息后按「调用按名入队、结果按名出队」配对
/// 并分配唯一 id。此前结果直接以函数名作 tool_use_id，与调用侧合成的
/// `{name}-call` 永不匹配，转 Anthropic 等强校验上游会因悬空 tool_result
/// 整请求 400；同名函数多次调用时合成 id 还会重复。
const FCALL_PENDING: &str = "__gemini_fcall__";
const FR_NAME_PREFIX: &str = "__gemini_fres__:";

/// finishReason → Core StopReason。
pub fn map_finish_reason(s: &str) -> Option<StopReason> {
    match s {
        "MAX_TOKENS" => Some(StopReason::MaxTokens),
        "SAFETY" | "RECITATION" | "PROHIBITED_CONTENT" | "BLOCKLIST" | "SPII" => {
            Some(StopReason::ContentFilter)
        }
        "STOP" | "LANGUAGE" | "OTHER" | "MALFORMED_FUNCTION_CALL" => Some(StopReason::EndTurn),
        _ => None,
    }
}

/// Core StopReason → Gemini finishReason。
pub fn unmap_stop_reason(r: Option<StopReason>) -> &'static str {
    match r {
        Some(StopReason::MaxTokens) => "MAX_TOKENS",
        Some(StopReason::ContentFilter) | Some(StopReason::Refusal) => "SAFETY",
        _ => "STOP",
    }
}

/// `generateContent` 路径（非流式）。
pub fn generate_path(version: &str, model: &str) -> String {
    format!(
        "/{}/models/{}:generateContent",
        version.trim_matches('/'),
        model
    )
}

/// `streamGenerateContent` 路径（流式，`alt=sse` 以标准 SSE 返回）。
pub fn stream_path(version: &str, model: &str) -> String {
    format!(
        "/{}/models/{}:streamGenerateContent?alt=sse",
        version.trim_matches('/'),
        model
    )
}

/// 提取内容块中的纯文本。
pub fn text_of(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

// ============================================================================
// Core → Gemini
// ============================================================================

/// 暂存签名并入前一 part（无签名时补写 thoughtSignature）；
/// 无可并入 part 时生成独立 `{"thought":true,"thoughtSignature":sig}`。
fn flush_pending_sig(parts: &mut Vec<Value>, pending: &mut Option<String>) {
    let Some(sig) = pending.take() else {
        return;
    };
    if let Some(last) = parts.last_mut() {
        if last.get("thoughtSignature").is_none() {
            last["thoughtSignature"] = json!(sig);
            return;
        }
    }
    parts.push(json!({ "thought": true, "thoughtSignature": sig }));
}

/// Core messages → Gemini `contents`（system 角色跳过，另走 systemInstruction）。
///
/// Gemini 的 `functionResponse` 按**函数名**关联结果，而 Core 的 `ToolResult` 只有
/// `tool_use_id`；故先扫描全部 `ToolUse` 建立 id→name 映射，再据此回填函数名。
pub fn core_to_contents(messages: &[Message]) -> Vec<Value> {
    let mut id_to_name: HashMap<&str, &str> = HashMap::new();
    for m in messages {
        for b in &m.content {
            if let ContentBlock::ToolUse { id, name, .. } = b {
                id_to_name.insert(id.as_str(), name.as_str());
            }
        }
    }

    let mut out = Vec::new();
    for m in messages {
        if m.role == Role::System {
            continue;
        }
        let role = match m.role {
            Role::Assistant => "model",
            // Tool 结果在 Gemini 里以 user 回合的 functionResponse part 呈现
            _ => "user",
        };
        let mut parts: Vec<Value> = Vec::new();
        // 「仅凭据」块（空文本 Reasoning{sig}）是 FC/text 签名跨协议的
        // 传输载体：紧邻 ToolUse 之前时迁移到 functionCall part（绑定目标），
        // 其余情况并入前一 part（末块/文本 part 签名原位），无位置可并时
        // 生成独立 thought part。
        let mut pending_sig: Option<String> = None;
        for b in &m.content {
            match b {
                ContentBlock::Text { text } => {
                    flush_pending_sig(&mut parts, &mut pending_sig);
                    parts.push(json!({ "text": text }));
                }
                ContentBlock::Image { data, media_type } => {
                    flush_pending_sig(&mut parts, &mut pending_sig);
                    if media_type != "url" && media_type != "file" {
                        parts.push(
                            json!({ "inlineData": { "mimeType": media_type, "data": data } }),
                        );
                    }
                }
                ContentBlock::Document {
                    source,
                    media_type,
                    data,
                    name,
                } => {
                    flush_pending_sig(&mut parts, &mut pending_sig);
                    match source {
                        // fileData 是 Gemini 侧文件引用的标准通道；displayName
                        // 承载文件名。
                        DocSource::Url => {
                            let mut fd = json!({ "fileUri": data, "mimeType": media_type });
                            if let Some(n) = name {
                                fd["displayName"] = json!(n);
                            }
                            parts.push(json!({ "fileData": fd }));
                        }
                        // base64 文档走 inlineData（mimeType 任意）；File 源
                        // （平台 file_id）跨协议不可解，不下发。
                        DocSource::Base64 => parts.push(
                            json!({ "inlineData": { "mimeType": media_type, "data": data } }),
                        ),
                        DocSource::Text => parts.push(json!({ "text": data })),
                        DocSource::File => {}
                    }
                }
                ContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    signature,
                    ..
                } => {
                    // 自带签名优先，否则消费紧邻的前置载体凭据
                    let sig = crate::adapters::untag_signature(
                        crate::adapters::SIG_GEMINI,
                        signature.as_deref(),
                    )
                    .map(String::from)
                    .or_else(|| pending_sig.take());
                    // 服务端代码执行还原为 executableCode 原生 part（区别于
                    // 普通 functionCall：该 part 不参与工具声明校验）。
                    if name == EXEC_CODE {
                        let mut part = json!({ "executableCode": input });
                        if let Some(sig) = sig.filter(|s| !s.is_empty()) {
                            part["thoughtSignature"] = json!(sig);
                        }
                        parts.push(part);
                        continue;
                    }
                    // id 一并下发：Gemini v1beta 支持 functionCall.id，
                    // 客户端原样回传历史时入站可直接按 id 精确配对。
                    let mut fc = json!({ "name": name, "args": input, "id": id });
                    if let Some(sig) = sig.filter(|s| !s.is_empty()) {
                        fc["thoughtSignature"] = json!(sig);
                    }
                    parts.push(json!({ "functionCall": fc }));
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    flush_pending_sig(&mut parts, &mut pending_sig);
                    // 未配对的占位 id（`__gemini_fres__:{name}`）剥前缀取函数名——
                    // 结果本就只有函数名语义，占位串不能作为 Gemini 函数名下发。
                    let name = id_to_name
                        .get(tool_use_id.as_str())
                        .copied()
                        .unwrap_or_else(|| {
                            tool_use_id
                                .strip_prefix(FR_NAME_PREFIX)
                                .unwrap_or(tool_use_id.as_str())
                        });
                    // 代码执行结果还原为 codeExecutionResult 原生 part；
                    // outcome 只有 ok/fail 粒度可从 is_error 还原。
                    if name == EXEC_CODE || tool_use_id.starts_with(EXEC_CODE) {
                        parts.push(json!({
                            "codeExecutionResult": {
                                "outcome": if *is_error { "OUTCOME_FAILED" } else { "OUTCOME_OK" },
                                "output": text_of(content),
                            }
                        }));
                        continue;
                    }
                    // is_error 无原生通道，写入 response 保留位（同协议回传
                    // 时读回；对模型而言是结果对象里的显式错误标记）。
                    let mut resp = json!({ "result": text_of(content) });
                    if *is_error {
                        resp["is_error"] = json!(true);
                    }
                    // 已配对的真实调用 id 一并下发（v1beta 支持），客户端原样
                    // 回传时入站可精确配对；未配对的占位/名串不作 id 下发。
                    let mut fr = json!({ "name": name, "response": resp });
                    if id_to_name.contains_key(tool_use_id.as_str()) {
                        fr["id"] = json!(tool_use_id);
                    }
                    parts.push(json!({ "functionResponse": fr }));
                    // functionResponse 只承载 JSON 文本：内嵌图片/文档提升为
                    // 同一消息内紧随的 inlineData/fileData part（Gemini 侧工具
                    // 媒体结果的标准通道；经尾部拆分最终落入下一 user 回合），
                    // 附一行来源标注保持与 functionCall 的关联——否则经
                    // text_of 静默丢弃后模型只看到空结果。
                    let mut hoisted = false;
                    for b in content {
                        let part = match b {
                            ContentBlock::Image { data, media_type } => {
                                if media_type == "url" || media_type == "file" {
                                    continue; // 与顶层分支同口径：引用形态不下发
                                }
                                json!({ "inlineData": { "mimeType": media_type, "data": data } })
                            }
                            ContentBlock::Document {
                                source,
                                media_type,
                                data,
                                name: doc_name,
                            } => match source {
                                DocSource::Url => {
                                    let mut fd = json!({ "fileUri": data, "mimeType": media_type });
                                    if let Some(n) = doc_name {
                                        fd["displayName"] = json!(n);
                                    }
                                    json!({ "fileData": fd })
                                }
                                DocSource::Base64 => {
                                    json!({ "inlineData": { "mimeType": media_type, "data": data } })
                                }
                                DocSource::Text => json!({ "text": data }),
                                DocSource::File => continue,
                            },
                            _ => continue,
                        };
                        if !hoisted {
                            parts.push(json!({
                                "text": format!("[media content returned by function call {name}]")
                            }));
                            hoisted = true;
                        }
                        parts.push(part);
                    }
                }
                ContentBlock::Reasoning {
                    text, signature, ..
                } => {
                    // 历史 thought 回传：Gemini 2.5 多轮 function calling 要求
                    // thoughtSignature 原样带回（缺失会被拒或退化）。
                    let sig = crate::adapters::untag_signature(
                        crate::adapters::SIG_GEMINI,
                        signature.as_deref(),
                    );
                    if text.is_empty() {
                        // 仅凭据载体：不独立成 part，留给后继 ToolUse 或前一 part
                        if let Some(s) = sig {
                            if !s.is_empty() {
                                pending_sig = Some(s.to_string());
                            }
                        }
                        continue;
                    }
                    flush_pending_sig(&mut parts, &mut pending_sig);
                    let mut part = json!({ "thought": true, "text": text });
                    if let Some(s) = sig.filter(|s| !s.is_empty()) {
                        part["thoughtSignature"] = json!(s);
                    }
                    parts.push(part);
                }
            }
        }
        flush_pending_sig(&mut parts, &mut pending_sig);
        if m.role == Role::Assistant {
            hoist_signature_to_first_function_call(&mut parts);
        }
        if !parts.is_empty() {
            // 连续 Tool 消息并入上一 user 回合：官方要求当前回合内所有
            // functionResponse 紧跟全部 functionCall。混入的非 FR part
            // （标注/媒体/提醒文本）由末尾的 split 统一收拢到后续回合，
            // 这里无需限制合并目标的内容形态。
            let merged = m.role == Role::Tool
                && out
                    .last()
                    .and_then(|c: &Value| c.get("role"))
                    .and_then(|r| r.as_str())
                    == Some("user");
            if merged {
                if let Some(arr) = out
                    .last_mut()
                    .and_then(|c| c.get_mut("parts"))
                    .and_then(|p| p.as_array_mut())
                {
                    arr.extend(parts);
                }
            } else {
                out.push(json!({ "role": role, "parts": parts }));
            }
        }
    }
    split_mixed_function_response_turns(&mut out);
    out
}

/// part 上承载 thoughtSignature 的两个位置：part 级（thought/text part，
/// 或 flush_pending_sig 落在 FC part 上的情形）与 functionCall 对象内。
fn part_signature(p: &Value) -> Option<&Value> {
    if let Some(s) = p.get("thoughtSignature") {
        Some(s)
    } else {
        p["functionCall"].get("thoughtSignature")
    }
}

/// gemini-4 代校验：当前轮次内每个模型步骤的**第一个 functionCall part
/// 必须携带 thought_signature**（官方布局：有 FC 时签名只挂在首个 FC 上，
/// 无 FC 时才落在最后一个 part）。历史经 Anthropic 等协议回传后签名常
/// 留在 thought/text part 上（这些协议的 tool_use 无签名字段，签名只能
/// 随 thinking 块走），或因块序异常落到后续 FC 上，导致 400
/// "Function call is missing a thought_signature"。这里把回合内找到
/// 的签名迁回首个 FC。
fn hoist_signature_to_first_function_call(parts: &mut [Value]) {
    let Some(first_fc) = parts
        .iter()
        .position(|p| p.get("functionCall").is_some())
    else {
        return;
    };
    if part_signature(&parts[first_fc]).is_some() {
        return;
    }
    // 取回合内首个签名（位置靠前的优先——thinking 块回传的标准位置在
    // FC 之前；仅在之前没有时才取之后 part 上的）
    let Some(src) = parts.iter().position(|p| part_signature(p).is_some()) else {
        return;
    };
    let sig = if parts[src].get("thoughtSignature").is_some() {
        parts[src]
            .as_object_mut()
            .and_then(|p| p.remove("thoughtSignature"))
    } else {
        parts[src]["functionCall"]
            .as_object_mut()
            .and_then(|f| f.remove("thoughtSignature"))
    };
    if let Some(sig) = sig {
        parts[first_fc]["functionCall"]["thoughtSignature"] = sig;
    }
}

/// 上游（gemini-4 代校验）要求含 functionResponse 的 user 回合**纯净**：
/// 混入 text/媒体 part（无论在 functionResponse 之前还是之后）都会触发
/// 400——实测尾随 text 报 "Requests ending with a model turn"，前置 text
/// 报 "missing thought_signature"（轮次边界判定被干扰）。将回合内全部
/// 非 functionResponse part 按原相对顺序移到紧随其后的新 user 回合，
/// 保持 functionCall → functionResponse 的相邻关系不被隔开。
fn split_mixed_function_response_turns(out: &mut Vec<Value>) {
    let mut i = 0;
    while i < out.len() {
        let is_user = out[i].get("role").and_then(Value::as_str) == Some("user");
        if !is_user {
            i += 1;
            continue;
        }
        let Some(parts) = out[i].get_mut("parts").and_then(Value::as_array_mut) else {
            i += 1;
            continue;
        };
        let is_fr = |p: &Value| p.get("functionResponse").is_some();
        if !parts.iter().any(is_fr) || parts.iter().all(is_fr) {
            i += 1;
            continue;
        }
        let mut frs: Vec<Value> = Vec::new();
        let mut others: Vec<Value> = Vec::new();
        for p in std::mem::take(parts) {
            if is_fr(&p) {
                frs.push(p);
            } else {
                others.push(p);
            }
        }
        *parts = frs;
        out.insert(i + 1, json!({ "role": "user", "parts": others }));
        i += 2;
    }
}

/// Core system（顶层 + system 角色消息）→ Gemini `systemInstruction`。
pub fn core_to_system_instruction(system: &[ContentBlock], messages: &[Message]) -> Option<Value> {
    let mut texts: Vec<&str> = Vec::new();
    for b in system {
        if let ContentBlock::Text { text } = b {
            texts.push(text);
        }
    }
    for m in messages {
        if m.role == Role::System {
            for b in &m.content {
                if let ContentBlock::Text { text } = b {
                    texts.push(text);
                }
            }
        }
    }
    if texts.is_empty() {
        None
    } else {
        Some(json!({ "parts": [{ "text": texts.join("\n\n") }] }))
    }
}

/// Gemini `Schema` 支持的字段（OpenAPI 3.0 子集）。JSON Schema 专有字段
/// （`$schema`、`additionalProperties`、`propertyNames`、`allOf` 等）会被上游
/// 以 400 `Cannot find field` 整体拒绝，必须在转换时剥除。
const SCHEMA_SUPPORTED_KEYS: &[&str] = &[
    "type", "format", "title", "description", "nullable", "enum", "items", "properties",
    "required", "anyOf", "propertyOrdering", "default", "example", "pattern", "minimum",
    "maximum", "minLength", "maxLength", "minItems", "maxItems", "minProperties",
    "maxProperties",
];

/// 剥除后会丢失约束的结构性字段（需记日志）；`$schema`/`additionalProperties`/
/// `propertyNames` 等仅作元数据或默认即期望行为，剥除属预期，不记。
const SCHEMA_STRUCTURAL_KEYS: &[&str] = &[
    "allOf", "oneOf", "not", "if", "then", "else", "const", "$ref", "$defs",
    "patternProperties", "dependentSchemas", "dependentRequired", "contains",
];

/// 递归清洗 JSON Schema 为 Gemini 可接受的 OpenAPI 子集：
/// - 白名单字段保留；结构性字段剥除时记入 `dropped`（约束丢失，调用方记日志），
///   元数据字段静默剥除；
/// - 数值型 `exclusiveMinimum`/`exclusiveMaximum`（JSON Schema 2020-12）放宽为
///   闭区间 `minimum`/`maximum`，与显式 `minimum`/`maximum` 并存时取更严格者
///   （布尔形态随 draft-07 的 `minimum`/`maximum` 一并保留，本身丢弃）；
/// - `type` 联合形态（`["string","null"]`）归一为首类型 + `nullable`，单独
///   `"null"` 转为 `nullable: true`（Gemini 的 type 是单值枚举）；
/// - 仅向 schema 位置（`properties` 的值、`items`、`anyOf` 元素）递归，
///   `default`/`example`/`enum` 中的实例数据原样保留。
///
/// `dropped` 收集被剥除的结构性字段与降级形态，供调用方记日志。
fn sanitize_schema_logged<'a>(schema: &'a Value, dropped: &mut Vec<String>) -> Value {
    let Value::Object(map) = schema else {
        return schema.clone();
    };
    let mut out = serde_json::Map::new();
    // 数值型 exclusive 边界先暂存，循环结束后与显式 minimum/maximum 合并取严
    let mut excl_min: Option<&'a Value> = None;
    let mut excl_max: Option<&'a Value> = None;
    for (k, v) in map {
        match k.as_str() {
            "exclusiveMinimum" => {
                if v.is_number() {
                    excl_min = Some(v);
                }
            }
            "exclusiveMaximum" => {
                if v.is_number() {
                    excl_max = Some(v);
                }
            }
            "type" => match v {
                Value::String(t) if t == "null" => {
                    out.insert("nullable".to_string(), json!(true));
                }
                Value::String(_) => {
                    out.insert(k.clone(), v.clone());
                }
                Value::Array(ts) => {
                    let mut non_null = ts.iter().filter_map(|t| t.as_str()).filter(|t| *t != "null");
                    if let Some(first) = non_null.next() {
                        if non_null.next().is_some() {
                            dropped.push("type(联合降级为首类型)".to_string());
                        }
                        out.insert("type".to_string(), json!(first));
                    }
                    if ts.iter().any(|t| t.as_str() == Some("null")) {
                        out.insert("nullable".to_string(), json!(true));
                    }
                }
                _ => {}
            },
            "properties" => match v.as_object() {
                Some(m) => {
                    let props = m
                        .iter()
                        .map(|(pk, pv)| (pk.clone(), sanitize_schema_logged(pv, dropped)))
                        .collect();
                    out.insert(k.clone(), Value::Object(props));
                }
                None => dropped.push("properties(非对象，已丢弃)".to_string()),
            },
            // tuple 形态（数组）Gemini 不支持：多元素约束无法表达，丢弃而非误留
            "items" if v.is_object() => {
                out.insert(k.clone(), sanitize_schema_logged(v, dropped));
            }
            "items" => {
                dropped.push("items(非单 schema，已丢弃)".to_string());
            }
            "anyOf" => match v.as_array() {
                Some(arr) => {
                    let cleaned = arr
                        .iter()
                        .map(|s| sanitize_schema_logged(s, dropped))
                        .collect();
                    out.insert(k.clone(), Value::Array(cleaned));
                }
                None => dropped.push("anyOf(非数组，已丢弃)".to_string()),
            },
            _ if SCHEMA_SUPPORTED_KEYS.contains(&k.as_str()) => {
                out.insert(k.clone(), v.clone());
            }
            _ if SCHEMA_STRUCTURAL_KEYS.contains(&k.as_str()) => {
                dropped.push(k.clone());
            }
            _ => {}
        }
    }
    // 合并 exclusive 边界：与显式 minimum/maximum 取更严格者（下界取大、上界
    // 取小）；比较经 f64，落盘保留原 JSON 数值类型（避免整数被写成 5.0）
    if let Some(eb) = excl_min {
        let stricter = match (out.get("minimum").and_then(|m| m.as_f64()), eb.as_f64()) {
            (Some(m), Some(e)) => e > m,
            (None, Some(_)) => true,
            _ => false,
        };
        if stricter {
            out.insert("minimum".to_string(), eb.clone());
        }
    }
    if let Some(eb) = excl_max {
        let stricter = match (out.get("maximum").and_then(|m| m.as_f64()), eb.as_f64()) {
            (Some(m), Some(e)) => e < m,
            (None, Some(_)) => true,
            _ => false,
        };
        if stricter {
            out.insert("maximum".to_string(), eb.clone());
        }
    }
    Value::Object(out)
}

/// Core tools → Gemini `tools`（单层 functionDeclarations 数组）。
pub fn core_to_tools(tools: &[Tool]) -> Option<Value> {
    if tools.is_empty() {
        return None;
    }
    let decls: Vec<Value> = tools
        .iter()
        .map(|t| {
            let mut dropped = Vec::new();
            let params = sanitize_schema_logged(&t.input_schema, &mut dropped);
            if !dropped.is_empty() {
                tracing::warn!(
                    tool = %t.name,
                    ?dropped,
                    "工具 schema 含 Gemini 不支持的结构性字段，已剥除（对应约束丢失）"
                );
            }
            // 非对象 parameters（如显式 null）归一为空对象 schema，与入站解析口径一致
            let params = if params.is_object() {
                params
            } else {
                json!({ "type": "object" })
            };
            let mut d = json!({ "name": t.name, "parameters": params });
            if let Some(desc) = &t.description {
                d["description"] = json!(desc);
            }
            d
        })
        .collect();
    Some(json!([{ "functionDeclarations": decls }]))
}

/// Core tool_choice → Gemini `toolConfig`。
pub fn core_to_tool_config(tc: Option<&ToolChoice>) -> Option<Value> {
    let choice = tc?;
    let mode = match choice {
        ToolChoice::Auto => "AUTO",
        ToolChoice::None => "NONE",
        ToolChoice::Required | ToolChoice::Tool { .. } => "ANY",
    };
    let mut config = json!({ "mode": mode });
    if let ToolChoice::Tool { name } = choice {
        config["allowedFunctionNames"] = json!([name]);
    }
    Some(json!({ "functionCallingConfig": config }))
}

/// Core 采样参数 → Gemini `generationConfig`（全空则返回 None）。
pub fn core_to_generation_config(req: &CoreRequest) -> Option<Value> {
    let mut gc = json!({});
    let obj = gc.as_object_mut().expect("gc is object");
    if let Some(mt) = req.max_tokens {
        obj.insert("maxOutputTokens".to_string(), json!(mt));
    }
    if let Some(t) = req.temperature {
        obj.insert("temperature".to_string(), json!(t));
    }
    if let Some(p) = req.top_p {
        obj.insert("topP".to_string(), json!(p));
    }
    if !req.stop.is_empty() {
        obj.insert("stopSequences".to_string(), json!(req.stop));
    }
    // 推理强度：现行 GenAI 接口统一为 thinkingLevel 档位（thinkingBudget 已
    // 退出）。只发 LOW/HIGH——官方文档中全模型的公共子集（gemini-3-pro-preview
    // 仅低/高两档，MEDIUM 会被 400 拒绝），更细粒度按方向就近归入
    if let Some(effort) = reasoning_effort(req) {
        let level = match effort.to_ascii_lowercase().as_str() {
            // OpenAI reasoning_effort=none 意为关闭思考；Gemini 侧最接近的
            // 形态是不下发 thinkingConfig（模型默认动态思考），绝不能反向放大
            "none" => None,
            "minimal" | "low" => Some("LOW"),
            "medium" | "high" | "xhigh" | "max" => Some("HIGH"),
            other => {
                tracing::warn!(effort = other, "未知推理强度档位，按 HIGH 下发");
                Some("HIGH")
            }
        };
        if let Some(level) = level {
            obj.insert(
                "thinkingConfig".to_string(),
                json!({ "thinkingLevel": level, "includeThoughts": true }),
            );
        }
    }
    if obj.is_empty() {
        None
    } else {
        Some(gc)
    }
}

/// Core 内容块 → Gemini 响应 candidate 的 parts（入口 from_core_response 用）。
///
/// 与 `core_to_contents` 同一套载体规则：仅凭据 Reasoning 前绑后继
/// ToolUse，其余并入前一 part。
pub fn core_to_parts(content: &[ContentBlock]) -> Vec<Value> {
    let mut parts: Vec<Value> = Vec::new();
    let mut pending_sig: Option<String> = None;
    for b in content {
        match b {
            ContentBlock::Text { text } => {
                flush_pending_sig(&mut parts, &mut pending_sig);
                parts.push(json!({ "text": text }));
            }
            ContentBlock::ToolUse {
                id,
                name,
                input,
                signature,
                ..
            } => {
                let sig = crate::adapters::untag_signature(
                    crate::adapters::SIG_GEMINI,
                    signature.as_deref(),
                )
                .map(String::from)
                .or_else(|| pending_sig.take());
                if name == EXEC_CODE {
                    let mut part = json!({ "executableCode": input });
                    if let Some(sig) = sig.filter(|s| !s.is_empty()) {
                        part["thoughtSignature"] = json!(sig);
                    }
                    parts.push(part);
                    continue;
                }
                let mut fc = json!({ "name": name, "args": input, "id": id });
                if let Some(sig) = sig.filter(|s| !s.is_empty()) {
                    fc["thoughtSignature"] = json!(sig);
                }
                parts.push(json!({ "functionCall": fc }))
            }
            ContentBlock::Image { data, media_type } => {
                flush_pending_sig(&mut parts, &mut pending_sig);
                if media_type != "url" && media_type != "file" {
                    parts.push(json!({ "inlineData": { "mimeType": media_type, "data": data } }));
                }
            }
            ContentBlock::Document {
                source,
                media_type,
                data,
                name,
            } => {
                flush_pending_sig(&mut parts, &mut pending_sig);
                match source {
                    DocSource::Url => {
                        let mut fd = json!({ "fileUri": data, "mimeType": media_type });
                        if let Some(n) = name {
                            fd["displayName"] = json!(n);
                        }
                        parts.push(json!({ "fileData": fd }));
                    }
                    DocSource::Base64 => parts
                        .push(json!({ "inlineData": { "mimeType": media_type, "data": data } })),
                    DocSource::Text => parts.push(json!({ "text": data })),
                    DocSource::File => {}
                }
            }
            ContentBlock::Reasoning {
                text, signature, ..
            } => {
                // thought part 回传：凭据（thoughtSignature）随 part 原样带回
                let sig = crate::adapters::untag_signature(
                    crate::adapters::SIG_GEMINI,
                    signature.as_deref(),
                );
                if text.is_empty() {
                    if let Some(s) = sig.filter(|s| !s.is_empty()) {
                        pending_sig = Some(s.to_string());
                    }
                    continue;
                }
                flush_pending_sig(&mut parts, &mut pending_sig);
                let mut part = json!({ "thought": true, "text": text });
                if let Some(s) = sig.filter(|s| !s.is_empty()) {
                    part["thoughtSignature"] = json!(s);
                }
                parts.push(part);
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => {
                flush_pending_sig(&mut parts, &mut pending_sig);
                // 响应侧的代码执行结果（model 回合内）还原原生 part；
                // 其余 ToolResult 在响应中不出现，维持丢弃。
                if tool_use_id.starts_with(EXEC_CODE) {
                    parts.push(json!({
                        "codeExecutionResult": {
                            "outcome": if *is_error { "OUTCOME_FAILED" } else { "OUTCOME_OK" },
                            "output": text_of(content),
                        }
                    }));
                }
            }
        }
    }
    flush_pending_sig(&mut parts, &mut pending_sig);
    parts
}

// ============================================================================
// Gemini → Core
// ============================================================================

/// 从 functionResponse.response 提取可读文本。
fn resp_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Object(m) => {
            for key in ["result", "output", "content"] {
                if let Some(s) = m.get(key).and_then(|x| x.as_str()) {
                    return s.to_string();
                }
            }
            v.to_string()
        }
        Value::Null => String::new(),
        _ => v.to_string(),
    }
}

/// 单个 Gemini part → 0..n 个 Core 内容块。
///
/// 凭据传输约定：part 级 `thoughtSignature` 打上 `gem:` 来源标记；签名绑定
/// 目标是「紧邻的后继 ToolUse / 前一 part」，而非任意 reasoning 块——
/// 故带签名的普通 text part 产出 `Text + 仅凭据载体 Reasoning`（后者由
/// `core_to_contents` 的 flush 规则还原回原位）。
pub fn part_to_blocks(p: &Value) -> Vec<ContentBlock> {
    let mut out = Vec::new();
    let signature = crate::adapters::tag_signature(
        crate::adapters::SIG_GEMINI,
        p.get("thoughtSignature")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string()),
    );
    let thought = p.get("thought").and_then(|v| v.as_bool()).unwrap_or(false);
    if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
        if !t.is_empty() {
            if thought {
                // thought part：明文 + thoughtSignature（回传凭据）→ Reasoning 块
                out.push(ContentBlock::Reasoning {
                    text: t.to_string(),
                    signature,
                    redacted: false,
                });
            } else {
                // 普通文本 part 带签名（官方：无 FC 时签名在最后一个 part）：
                // 文本归文本，凭据作载体紧随——回传时并入前一 part 还原
                out.push(ContentBlock::text(t));
                if signature.is_some() {
                    out.push(ContentBlock::Reasoning {
                        text: String::new(),
                        signature,
                        redacted: false,
                    });
                }
            }
        } else if signature.is_some() {
            // 无文本但带凭据的 part：凭据不能丢，否则多轮回传缺失
            out.push(ContentBlock::Reasoning {
                text: String::new(),
                signature,
                redacted: false,
            });
        }
    } else if p.get("functionCall").is_none() && p.get("executableCode").is_none() {
        // 非文本且非调用类 part 带凭据：凭据不能丢；
        // functionCall/executableCode part 的凭据归 ToolUse 块（见下），不在此处理
        if signature.is_some() {
            out.push(ContentBlock::Reasoning {
                text: String::new(),
                signature,
                redacted: false,
            });
        }
    }
    if let Some(fc) = p.get("functionCall") {
        let name = fc
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        // Gemini 的 functionCall 不一定带 id；缺失时用占位值，由
        // contents_to_core 统一分配唯一 id 并与 functionResponse 按名配对。
        let id = fc
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| FCALL_PENDING.to_string());
        let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
        // 加密 CoT 凭据：与 function call 强绑定的 thoughtSignature（part 级字段，
        // 官方形态在 part 上；部分实现放在 functionCall 内，均兼容）。
        // ToolUse.signature 直接携带 + 前置载体块回退（无凭据位的协议靠载体还原）。
        let signature = crate::adapters::tag_signature(
            crate::adapters::SIG_GEMINI,
            p.get("thoughtSignature")
                .or_else(|| fc.get("thoughtSignature"))
                .and_then(|v| v.as_str())
                .map(String::from),
        );
        if let Some(sig) = &signature {
            out.push(ContentBlock::Reasoning {
                text: String::new(),
                signature: Some(sig.clone()),
                redacted: false,
            });
        }
        out.push(ContentBlock::ToolUse {
            id,
            item_id: None,
            name,
            namespace: None,
            input: args,
            signature,
        });
    }
    if let Some(fr) = p.get("functionResponse") {
        let name = fr
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let resp = fr.get("response").cloned().unwrap_or(Value::Null);
        // id 缺省时记 `{FR_NAME_PREFIX}{name}` 占位——结果本身只有函数名，
        // 配对由 contents_to_core 按序完成；同协议回传时 id_to_name 解析
        // 出的是占位串，须在此剥掉前缀还原为函数名语义。
        let tool_use_id = fr
            .get("id")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("{FR_NAME_PREFIX}{name}"));
        out.push(ContentBlock::ToolResult {
            tool_use_id,
            content: vec![ContentBlock::text(resp_to_text(&resp))],
            // response.is_error 是本站出站侧写入的保留位（Gemini 无原生
            // 错误语义），回传时读回保真；外部实现不会带此键。
            is_error: resp
                .get("is_error")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        });
    }
    if let Some(inline) = p.get("inlineData") {
        let data = inline
            .get("data")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let mt = inline
            .get("mimeType")
            .and_then(|v| v.as_str())
            .unwrap_or("image/png")
            .to_string();
        out.push(ContentBlock::Image {
            data,
            media_type: mt,
        });
    }
    if let Some(fd) = p.get("fileData") {
        // Files API 引用 → Document（Uri 源）：此前整块丢弃。displayName
        // 承载文件名；mimeType 保留真实类型供回传还原。
        let uri = fd
            .get("fileUri")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if !uri.is_empty() {
            out.push(ContentBlock::Document {
                source: DocSource::Url,
                media_type: fd
                    .get("mimeType")
                    .and_then(|v| v.as_str())
                    .unwrap_or("application/octet-stream")
                    .to_string(),
                data: uri.to_string(),
                name: fd
                    .get("displayName")
                    .and_then(|v| v.as_str())
                    .map(String::from),
            });
        }
    }
    // 服务端代码执行对：executableCode（待执行代码）→ ToolUse、
    // codeExecutionResult（执行输出）→ ToolResult，均以 EXEC_CODE 占位名
    // 承载；tool_use_id 用占位值，由 contents_to_core 统一配对编号。
    if let Some(ec) = p.get("executableCode") {
        let signature = crate::adapters::tag_signature(
            crate::adapters::SIG_GEMINI,
            p.get("thoughtSignature")
                .or_else(|| ec.get("thoughtSignature"))
                .and_then(|v| v.as_str())
                .map(String::from),
        );
        if let Some(sig) = &signature {
            out.push(ContentBlock::Reasoning {
                text: String::new(),
                signature: Some(sig.clone()),
                redacted: false,
            });
        }
        out.push(ContentBlock::ToolUse {
            id: EXEC_CODE.to_string(),
            item_id: None,
            name: EXEC_CODE.to_string(),
            namespace: None,
            input: ec.clone(),
            signature,
        });
    }
    if let Some(er) = p.get("codeExecutionResult") {
        let outcome = er
            .get("outcome")
            .and_then(|v| v.as_str())
            .unwrap_or("OUTCOME_OK");
        let output = er
            .get("output")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        out.push(ContentBlock::ToolResult {
            tool_use_id: EXEC_CODE.to_string(),
            content: vec![ContentBlock::text(output)],
            is_error: outcome != "OUTCOME_OK",
        });
    }
    out
}

/// Gemini `contents` → Core messages。
pub fn contents_to_core(contents: &[Value]) -> Vec<Message> {
    let mut msgs: Vec<Message> = contents
        .iter()
        .map(|c| {
            let role = match c.get("role").and_then(|r| r.as_str()) {
                Some("model") => Role::Assistant,
                _ => Role::User,
            };
            let parts = c
                .get("parts")
                .and_then(|p| p.as_array())
                .cloned()
                .unwrap_or_default();
            let mut content = Vec::new();
            for p in &parts {
                content.extend(part_to_blocks(p));
            }
            // executableCode/codeExecutionResult 成对出现：part_to_blocks 里
            // 用占位 id，此处按序分配唯一 id 并配对——跨上游（Anthropic 等）
            // 要求 tool_use id 全局唯一且结果必须回指对应调用。
            let mut seq = 0usize;
            let mut pending: Option<String> = None;
            for b in &mut content {
                match b {
                    ContentBlock::ToolUse { id, name, .. } if name == EXEC_CODE => {
                        let assigned = format!("{EXEC_CODE}-{seq}");
                        seq += 1;
                        *id = assigned.clone();
                        pending = Some(assigned);
                    }
                    ContentBlock::ToolResult { tool_use_id, .. } if tool_use_id == EXEC_CODE => {
                        if let Some(id) = pending.take() {
                            *tool_use_id = id;
                        }
                    }
                    _ => {}
                }
            }
            Message {
                role,
                content,
                ext: Default::default(),
            }
        })
        .collect();

    // functionCall/functionResponse 配对：调用无 id 时用占位值，此处分配
    // 唯一 id 并按函数名入队；结果无 id 时（`{FR_NAME_PREFIX}{name}`）按名
    // 出队回指对应调用——Gemini 语义即 user 回合的 functionResponse 按序
    // 对应前序 model 回合的同名 functionCall。带真实 id 的调用同样入队，
    // 使缺 id 的结果仍能按名配对；悬空结果保留函数名（旧行为， outbound
    // 剥前缀还原）。跨上游（Anthropic 等）要求结果 id 回指真实调用 id，
    // 不配对会以「找不到对应 tool_use」整请求 400。
    let mut seq = 0usize;
    let mut pending: HashMap<String, std::collections::VecDeque<String>> = HashMap::new();
    for m in &mut msgs {
        for b in &mut m.content {
            match b {
                ContentBlock::ToolUse { id, name, .. } => {
                    if *id == FCALL_PENDING {
                        *id = format!("fcall-{seq}");
                        seq += 1;
                    }
                    pending
                        .entry(name.clone())
                        .or_default()
                        .push_back(id.clone());
                }
                ContentBlock::ToolResult { tool_use_id, .. } => {
                    if let Some(name) = tool_use_id.strip_prefix(FR_NAME_PREFIX) {
                        let matched = pending.get_mut(name).and_then(|q| q.pop_front());
                        *tool_use_id = matched.unwrap_or_else(|| name.to_string());
                    }
                }
                _ => {}
            }
        }
    }
    msgs
}

/// Gemini `systemInstruction` → Core system 内容块。
pub fn system_instruction_to_core(si: Option<&Value>) -> Vec<ContentBlock> {
    let mut out = Vec::new();
    match si {
        Some(Value::String(s)) => {
            if !s.is_empty() {
                out.push(ContentBlock::text(s.clone()));
            }
        }
        Some(v) => {
            if let Some(parts) = v.get("parts").and_then(|p| p.as_array()) {
                for p in parts {
                    if let Some(t) = p.get("text").and_then(|x| x.as_str()) {
                        if !t.is_empty() {
                            out.push(ContentBlock::text(t));
                        }
                    }
                }
            }
        }
        None => {}
    }
    out
}

/// Gemini `tools` → Core tools。
pub fn tools_to_core(tools: &[Value]) -> Vec<Tool> {
    let mut out = Vec::new();
    for t in tools {
        if let Some(decls) = t.get("functionDeclarations").and_then(|d| d.as_array()) {
            for d in decls {
                let Some(name) = d.get("name").and_then(|v| v.as_str()) else {
                    continue;
                };
                out.push(Tool {
                    name: name.to_string(),
                    description: d
                        .get("description")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    input_schema: d
                        .get("parameters")
                        .cloned()
                        .unwrap_or_else(|| json!({ "type": "object" })),
                    ext: Default::default(),
                });
            }
        }
    }
    out
}

/// Gemini `toolConfig` → Core tool_choice。
pub fn tool_config_to_core(tc: Option<&Value>) -> Option<ToolChoice> {
    let fcc = tc?.get("functionCallingConfig")?;
    match fcc.get("mode").and_then(|m| m.as_str()) {
        Some("AUTO") => Some(ToolChoice::Auto),
        Some("NONE") => Some(ToolChoice::None),
        Some("ANY") => {
            if let Some(names) = fcc.get("allowedFunctionNames").and_then(|n| n.as_array()) {
                if names.len() == 1 {
                    if let Some(n) = names[0].as_str() {
                        return Some(ToolChoice::Tool {
                            name: n.to_string(),
                        });
                    }
                }
            }
            Some(ToolChoice::Required)
        }
        _ => None,
    }
}

/// Gemini 响应 candidate → Core 内容块 + finishReason。
pub fn candidate_to_core(cand: &Value) -> (Vec<ContentBlock>, Option<StopReason>) {
    let mut content = Vec::new();
    if let Some(parts) = cand
        .get("content")
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
    {
        for p in parts {
            content.extend(part_to_blocks(p));
        }
    }
    let finish = cand
        .get("finishReason")
        .and_then(|v| v.as_str())
        .and_then(map_finish_reason);
    (content, finish)
}

/// Gemini `usageMetadata` → Core Usage。
///
/// Core 不变量：`output_tokens` 含 reasoning——`candidatesTokenCount` 不含
/// `thoughtsTokenCount`（OpenAI 的 `completion_tokens` 则含），必须补上，否则
/// Gemini 上游的输出 token 系统性少算、计价漏掉按输出价计费的思考 token。
/// `toolUsePromptTokenCount` 独立于 `promptTokenCount`（官方 totalTokenCount =
/// prompt + toolUse + candidates + thoughts），按输入价计费，并入 input。
pub fn usage_from_gemini(v: &Value) -> Usage {
    let thoughts = v
        .get("thoughtsTokenCount")
        .and_then(|x| x.as_u64())
        .unwrap_or(0) as u32;
    Usage {
        input_tokens: v
            .get("promptTokenCount")
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as u32
            + v.get("toolUsePromptTokenCount")
                .and_then(|x| x.as_u64())
                .unwrap_or(0) as u32,
        output_tokens: v
            .get("candidatesTokenCount")
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as u32
            + thoughts,
        cache_read_tokens: v
            .get("cachedContentTokenCount")
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as u32,
        cache_write_tokens: 0,
        reasoning_tokens: thoughts,
    }
}

pub fn usage_object(u: &Usage) -> Value {
    json!({
        "promptTokenCount": u.input_tokens,
        "candidatesTokenCount": u.output_tokens,
        "totalTokenCount": u.input_tokens + u.output_tokens,
        "thoughtsTokenCount": u.reasoning_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// thought part 的 thoughtSignature 凭据 round-trip。
    #[test]
    fn thought_signature_roundtrips() {
        let part = json!({ "text": "thinking...", "thought": true, "thoughtSignature": "SIG" });
        let blocks = part_to_blocks(&part);
        match &blocks[0] {
            ContentBlock::Reasoning {
                text,
                signature: Some(sig),
                ..
            } => {
                assert_eq!(text, "thinking...");
                assert_eq!(sig, "gem:SIG", "凭据带来源标记，出站时还原");
            }
            other => panic!("expected reasoning, got {other:?}"),
        }
        // 历史回传：thoughtSignature 原样带回
        let msgs = vec![Message {
            role: Role::Assistant,
            content: blocks,
            ext: Default::default(),
        }];
        let contents = core_to_contents(&msgs);
        let p = &contents[0]["parts"][0];
        assert_eq!(p["thought"], true);
        assert_eq!(p["thoughtSignature"], "SIG");
        assert_eq!(p["text"], "thinking...");
    }

    #[test]
    fn maps_finish_reasons() {
        assert_eq!(map_finish_reason("STOP"), Some(StopReason::EndTurn));
        assert_eq!(map_finish_reason("MAX_TOKENS"), Some(StopReason::MaxTokens));
        assert_eq!(map_finish_reason("SAFETY"), Some(StopReason::ContentFilter));
        assert_eq!(unmap_stop_reason(Some(StopReason::MaxTokens)), "MAX_TOKENS");
        assert_eq!(unmap_stop_reason(Some(StopReason::ToolUse)), "STOP");
    }

    #[test]
    fn builds_paths() {
        assert_eq!(
            generate_path("v1beta", "gemini-2.0-flash"),
            "/v1beta/models/gemini-2.0-flash:generateContent"
        );
        assert!(stream_path("v1", "g").contains(":streamGenerateContent?alt=sse"));
    }

    #[test]
    fn roundtrips_tool_result_by_name() {
        // assistant 发起 get_time（id=c1），随后 user 回合回传结果（tool_use_id=c1）
        let msgs = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    item_id: None,
                    name: "get_time".into(),
                    namespace: None,
                    input: json!({"tz": "UTC"}),
                    signature: None,
                }],
                ext: Default::default(),
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "c1".into(),
                    content: vec![ContentBlock::text("12:00")],
                    is_error: false,
                }],
                ext: Default::default(),
            },
        ];
        let contents = core_to_contents(&msgs);
        assert_eq!(contents[0]["role"], "model");
        assert_eq!(contents[0]["parts"][0]["functionCall"]["name"], "get_time");
        assert_eq!(contents[1]["role"], "user");
        // functionResponse 按名关联，而非 id
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["name"],
            "get_time"
        );
        assert_eq!(
            contents[1]["parts"][0]["functionResponse"]["response"]["result"],
            "12:00"
        );
    }

    #[test]
    fn parses_candidate_with_function_call() {
        let cand = json!({
            "content": { "role": "model", "parts": [{ "functionCall": { "name": "get_time", "args": {"tz":"UTC"} } }] },
            "finishReason": "STOP"
        });
        let (content, finish) = candidate_to_core(&cand);
        assert_eq!(finish, Some(StopReason::EndTurn));
        match &content[0] {
            ContentBlock::ToolUse {
                name, input, id, ..
            } => {
                assert_eq!(name, "get_time");
                assert_eq!(input["tz"], "UTC");
                assert!(!id.is_empty(), "缺失 id 时应合成");
            }
            other => panic!("expected tool_use, got {other:?}"),
        }
    }

    /// 加密 CoT 凭据与 function call 强绑定：functionCall part 的
    /// thoughtSignature → ToolUse.signature，历史回传时还原到 part。
    #[test]
    fn function_call_thought_signature_roundtrips() {
        let cand = json!({
            "content": { "role": "model", "parts": [{
                "functionCall": { "name": "get_time", "args": {"tz": "UTC"} },
                "thoughtSignature": "SIG"
            }] },
            "finishReason": "STOP"
        });
        let (content, _) = candidate_to_core(&cand);
        // 载体在前（供无 ToolUse 凭据位的协议还原），ToolUse 自带签名
        match &content[0] {
            ContentBlock::Reasoning {
                text,
                signature: Some(sig),
                ..
            } => {
                assert!(text.is_empty());
                assert_eq!(sig, "gem:SIG");
            }
            other => panic!("expected signature carrier, got {other:?}"),
        }
        match &content[1] {
            ContentBlock::ToolUse {
                signature: Some(sig),
                ..
            } => assert_eq!(sig, "gem:SIG"),
            other => panic!("expected tool_use with signature, got {other:?}"),
        }

        // 历史回传：凭据随 functionCall part 原样带回
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "c1".into(),
                item_id: None,
                name: "get_time".into(),
                namespace: None,
                input: json!({"tz": "UTC"}),
                signature: Some("SIG".into()),
            }],
            ext: Default::default(),
        };
        let contents = core_to_contents(&[
            Message {
                role: Role::User,
                content: vec![ContentBlock::text("q")],
                ext: Default::default(),
            },
            msg,
        ]);
        assert_eq!(
            contents[1]["parts"][0]["functionCall"]["thoughtSignature"],
            "SIG"
        );
    }

    /// 普通文本 part 携带 thoughtSignature（官方：无 FC 时签名在最后一个
    /// part）：text 归 Text 块、凭据拆出为紧邻的仅凭据载体 Reasoning——
    /// 不再误挂载到 reasoning 槽位，回传时载体并入前一 part 还原。
    #[test]
    fn text_part_signature_detaches_to_carrier() {
        let blocks = part_to_blocks(&json!({"text": "Hi", "thoughtSignature": "S"}));
        assert_eq!(blocks.len(), 2);
        match &blocks[0] {
            ContentBlock::Text { text } => assert_eq!(text, "Hi"),
            other => panic!("文本归 Text 块, got {other:?}"),
        }
        match &blocks[1] {
            ContentBlock::Reasoning {
                text, signature, ..
            } => {
                assert!(text.is_empty());
                assert_eq!(signature.as_deref(), Some("gem:S"));
            }
            other => panic!("签名应拆出为载体块, got {other:?}"),
        }

        // 回传还原：载体并入前一 part
        let contents = core_to_contents(&[Message {
            role: Role::Assistant,
            content: blocks,
            ext: Default::default(),
        }]);
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1, "载体不额外占 part");
        assert_eq!(parts[0]["text"], "Hi");
        assert_eq!(parts[0]["thoughtSignature"], "S");
    }

    /// gemini-4 代校验：模型回合含 functionCall 时，thought_signature 必须
    /// 挂在首个 functionCall 上——历史经 Anthropic 回传后签名常留在
    /// thinking 对应的 thought part 上，须迁移，否则 400。
    #[test]
    fn signature_on_thought_part_hoists_to_first_function_call() {
        let tool_use = |id: &str| ContentBlock::ToolUse {
            id: id.into(),
            item_id: None,
            name: "Bash".into(),
            namespace: None,
            input: json!({}),
            signature: None,
        };
        let msgs = vec![
            Message::text(Role::User, "q"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Reasoning {
                        text: "thinking…".into(),
                        signature: Some("gem:SIG".into()),
                        redacted: false,
                    },
                    tool_use("c1"),
                    tool_use("c2"),
                ],
                ext: Default::default(),
            },
        ];
        let contents = core_to_contents(&msgs);
        let parts = contents[1]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 3);
        assert!(
            parts[0].get("thoughtSignature").is_none(),
            "签名应迁离 thought part: {parts:?}"
        );
        assert_eq!(
            parts[1]["functionCall"]["thoughtSignature"],
            "SIG",
            "首个 functionCall 应携带签名"
        );
        assert!(
            parts[2]["functionCall"].get("thoughtSignature").is_none(),
            "并行调用的后续 FC 不带签名"
        );
    }

    /// 首个 functionCall 自带签名时不做迁移（其它 part 的签名保持原位）。
    #[test]
    fn existing_function_call_signature_is_not_rewritten() {
        let msgs = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Reasoning {
                    text: "thinking…".into(),
                    signature: Some("gem:THOUGHT_SIG".into()),
                    redacted: false,
                },
                ContentBlock::ToolUse {
                    id: "c1".into(),
                    item_id: None,
                    name: "Bash".into(),
                    namespace: None,
                    input: json!({}),
                    signature: Some("gem:FC_SIG".into()),
                },
            ],
            ext: Default::default(),
        }];
        let contents = core_to_contents(&msgs);
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts[0]["thoughtSignature"], "THOUGHT_SIG");
        assert_eq!(parts[1]["functionCall"]["thoughtSignature"], "FC_SIG");
    }

    /// 审查回归：签名落在后续 FC 上（仅凭据载体错位到第二个 ToolUse）
    /// 时仍须迁回首个 FC——否则首个 FC 无签名，触发上游 400。
    #[test]
    fn signature_on_later_function_call_moves_to_first() {
        let tool_use = |id: &str| ContentBlock::ToolUse {
            id: id.into(),
            item_id: None,
            name: "Bash".into(),
            namespace: None,
            input: json!({}),
            signature: None,
        };
        let msgs = vec![Message {
            role: Role::Assistant,
            content: vec![
                tool_use("c1"),
                ContentBlock::Reasoning {
                    text: String::new(),
                    signature: Some("gem:SIG".into()),
                    redacted: false,
                },
                tool_use("c2"),
            ],
            ext: Default::default(),
        }];
        let contents = core_to_contents(&msgs);
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(
            parts[0]["functionCall"]["thoughtSignature"],
            "SIG",
            "首个 FC 应拿到签名: {parts:?}"
        );
        assert!(
            parts[1]["functionCall"].get("thoughtSignature").is_none()
                && parts[1].get("thoughtSignature").is_none(),
            "签名不得重复滞留: {parts:?}"
        );
    }

    /// 审查回归：并行工具结果带媒体时，连续 Tool 消息先合并、再由末尾
    /// 拆分收拢——functionResponse 回合纯净且与 functionCall 回合相邻，
    /// 媒体 part 统一进入紧随其后的 user 回合。
    #[test]
    fn parallel_tool_results_with_media_merge_then_split() {
        let tool_use = |id: &str| ContentBlock::ToolUse {
            id: id.into(),
            item_id: None,
            name: "Bash".into(),
            namespace: None,
            input: json!({}),
            signature: None,
        };
        let msgs = vec![
            Message {
                role: Role::Assistant,
                content: vec![tool_use("c1"), tool_use("c2")],
                ext: Default::default(),
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "c1".into(),
                    content: vec![
                        ContentBlock::text("shot"),
                        ContentBlock::Image {
                            data: "AAA".into(),
                            media_type: "image/png".into(),
                        },
                    ],
                    is_error: false,
                }],
                ext: Default::default(),
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "c2".into(),
                    content: vec![ContentBlock::text("ok")],
                    is_error: false,
                }],
                ext: Default::default(),
            },
        ];
        let contents = core_to_contents(&msgs);
        assert_eq!(contents.len(), 3, "FR 回合 + 媒体回合: {contents:?}");
        let fr_parts = contents[1]["parts"].as_array().unwrap();
        assert_eq!(
            fr_parts.iter().filter(|p| p.get("functionResponse").is_some()).count(),
            2,
            "两个 functionResponse 必须在同一纯净回合: {fr_parts:?}"
        );
        assert_eq!(fr_parts.len(), 2);
        let media_parts = contents[2]["parts"].as_array().unwrap();
        assert_eq!(media_parts[0]["text"], "[media content returned by function call Bash]");
        assert_eq!(media_parts[1]["inlineData"]["data"], "AAA");
    }

    /// 连续 Tool 消息合并为单个 user content（官方：FR 紧跟全部 FC，
    /// 拆多个 user 回合可能触发严格校验 400）。
    #[test]
    fn consecutive_tool_results_merge_into_one_user_content() {
        let msgs = vec![
            Message::text(Role::User, "q"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::ToolUse {
                        id: "c1".into(),
                        item_id: None,
                        name: "a".into(),
                        namespace: None,
                        input: json!({}),
                        signature: None,
                    },
                    ContentBlock::ToolUse {
                        id: "c2".into(),
                        item_id: None,
                        name: "b".into(),
                        namespace: None,
                        input: json!({}),
                        signature: None,
                    },
                ],
                ext: Default::default(),
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "c1".into(),
                    content: vec![ContentBlock::text("r1")],
                    is_error: false,
                }],
                ext: Default::default(),
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "c2".into(),
                    content: vec![ContentBlock::text("r2")],
                    is_error: false,
                }],
                ext: Default::default(),
            },
        ];
        let contents = core_to_contents(&msgs);
        assert_eq!(contents.len(), 3, "两个 Tool 结果应合并为一个 user 回合");
        assert_eq!(contents[2]["role"], "user");
        assert_eq!(contents[2]["parts"].as_array().unwrap().len(), 2);
    }

    /// 回归：tool_result 之后同消息再跟 text（如 harness 注入的提醒）时，
    /// functionResponse 独占 user 回合、尾随 text 拆为紧随的独立 user
    /// 回合——同回合混排会触发上游 400 "Requests ending with a model turn"。
    #[test]
    fn trailing_text_after_tool_results_splits_into_new_user_turn() {
        let tool_use = |id: &str| ContentBlock::ToolUse {
            id: id.into(),
            item_id: None,
            name: "Bash".into(),
            namespace: None,
            input: json!({}),
            signature: None,
        };
        let tool_result = |id: &str| ContentBlock::ToolResult {
            tool_use_id: id.into(),
            content: vec![ContentBlock::text("ok")],
            is_error: false,
        };
        let msgs = vec![
            Message::text(Role::User, "q"),
            Message {
                role: Role::Assistant,
                content: vec![tool_use("c1"), tool_use("c2")],
                ext: Default::default(),
            },
            Message {
                role: Role::User,
                content: vec![
                    tool_result("c1"),
                    tool_result("c2"),
                    ContentBlock::text("reminder"),
                ],
                ext: Default::default(),
            },
        ];
        let contents = core_to_contents(&msgs);
        assert_eq!(contents.len(), 4, "尾随 text 应拆为独立回合: {contents:?}");
        let fr_parts = contents[2]["parts"].as_array().unwrap();
        assert_eq!(contents[2]["role"], "user");
        assert!(
            fr_parts.iter().all(|p| p.get("functionResponse").is_some()),
            "functionResponse 回合不得混入其他 part: {fr_parts:?}"
        );
        assert_eq!(fr_parts.len(), 2);
        assert_eq!(contents[3]["role"], "user");
        assert_eq!(
            contents[3]["parts"].as_array().unwrap(),
            &vec![json!({ "text": "reminder" })]
        );
    }

    /// 回归：text 位于 tool_result 之前的混合回合同样拆分——非
    /// functionResponse part 统一移到后续独立 user 回合，避免干扰上游的
    /// 轮次边界判定（实测报 400 "missing thought_signature"）。
    #[test]
    fn leading_text_before_tool_results_moves_after_function_response_turn() {
        let msgs = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    item_id: None,
                    name: "Bash".into(),
                    namespace: None,
                    input: json!({}),
                    signature: None,
                }],
                ext: Default::default(),
            },
            Message {
                role: Role::User,
                content: vec![
                    ContentBlock::text("note"),
                    ContentBlock::ToolResult {
                        tool_use_id: "c1".into(),
                        content: vec![ContentBlock::text("ok")],
                        is_error: false,
                    },
                ],
                ext: Default::default(),
            },
        ];
        let contents = core_to_contents(&msgs);
        assert_eq!(contents.len(), 3, "前置 text 应移到 FR 回合之后: {contents:?}");
        let fr_parts = contents[1]["parts"].as_array().unwrap();
        assert!(
            fr_parts.iter().all(|p| p.get("functionResponse").is_some()),
            "FR 回合必须纯净: {fr_parts:?}"
        );
        assert_eq!(
            contents[2]["parts"].as_array().unwrap(),
            &vec![json!({ "text": "note" })]
        );
    }

    /// 载体凭据迁移：空文本 Reasoning{sig} 紧邻 ToolUse 之前 → 签名落到
    /// functionCall part（非 Gemini 入口回传的标准形态）；异源凭据不迁移。
    #[test]
    fn signature_carrier_attaches_to_next_tool_use() {
        let msgs = vec![
            Message::text(Role::User, "q"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Reasoning {
                        text: String::new(),
                        signature: Some("gem:SIG".into()),
                        redacted: false,
                    },
                    ContentBlock::ToolUse {
                        id: "c1".into(),
                        item_id: None,
                        name: "f".into(),
                        namespace: None,
                        input: json!({}),
                        signature: None,
                    },
                ],
                ext: Default::default(),
            },
        ];
        let contents = core_to_contents(&msgs);
        let parts = contents[1]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1, "载体不独立成 part");
        assert_eq!(parts[0]["functionCall"]["thoughtSignature"], "SIG");

        // 异源凭据（Anthropic 签名）不得迁移——透传只会让上游 400
        let mut foreign = msgs.clone();
        if let Some(ContentBlock::Reasoning { signature, .. }) = foreign[1].content.get_mut(0) {
            *signature = Some("ant:FOREIGN".into());
        }
        let contents = core_to_contents(&foreign);
        assert!(contents[1]["parts"][0]["functionCall"]
            .get("thoughtSignature")
            .is_none());
    }

    /// 回归：ToolResult 内嵌图片不得经 `text_of` 丢弃成空 functionResponse——
    /// functionResponse 只承载 JSON 文本，图片提升为紧随 user 回合的
    /// inlineData part（Gemini 侧工具图片结果的标准通道），附来源标注；
    /// url 形态与顶层 Image 分支同口径跳过。
    #[test]
    fn tool_result_images_hoist_as_inline_data() {
        let msgs = vec![
            Message::text(Role::User, "q"),
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "c1".into(),
                    item_id: None,
                    name: "read_file".into(),
                    namespace: None,
                    input: json!({ "path": "a.png" }),
                    signature: None,
                }],
                ext: Default::default(),
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "c1".into(),
                    content: vec![
                        ContentBlock::text("image below"),
                        ContentBlock::Image {
                            data: "AAA".into(),
                            media_type: "image/png".into(),
                        },
                        ContentBlock::Image {
                            data: "https://example.com/x.png".into(),
                            media_type: "url".into(),
                        },
                        ContentBlock::Image {
                            data: "BBB".into(),
                            media_type: "image/jpeg".into(),
                        },
                    ],
                    is_error: false,
                }],
                ext: Default::default(),
            },
        ];
        let contents = core_to_contents(&msgs);
        // functionResponse 独占回合；标注与媒体 part 拆入紧随的 user 回合
        // （gemini-4 代校验拒绝 functionResponse 后同回合再跟其他 part）
        assert_eq!(contents.len(), 4);
        let fr_parts = contents[2]["parts"].as_array().unwrap();
        assert_eq!(
            fr_parts[0],
            json!({ "functionResponse": { "name": "read_file", "response": { "result": "image below" }, "id": "c1" } })
        );
        assert_eq!(fr_parts.len(), 1);
        let media_parts = contents[3]["parts"].as_array().unwrap();
        assert_eq!(contents[3]["role"], "user");
        assert_eq!(
            media_parts[0],
            json!({ "text": "[media content returned by function call read_file]" })
        );
        assert_eq!(
            media_parts[1],
            json!({ "inlineData": { "mimeType": "image/png", "data": "AAA" } })
        );
        assert_eq!(
            media_parts[2],
            json!({ "inlineData": { "mimeType": "image/jpeg", "data": "BBB" } })
        );
        assert_eq!(media_parts.len(), 3, "url 形态图片不下发: {media_parts:?}");
    }

    /// 回归：fileData part（Files API 引用）不得静默丢弃——映射为 Uri 源
    /// Document（mimeType/displayName 保真），回传 Gemini 还原 fileData。
    #[test]
    fn file_data_part_maps_to_document() {
        let uri = "https://generativelanguage.googleapis.com/v1beta/files/abc";
        let blocks = part_to_blocks(&json!({
            "fileData": { "fileUri": uri, "mimeType": "application/pdf", "displayName": "s.pdf" }
        }));
        let ContentBlock::Document {
            source,
            media_type,
            data,
            name,
        } = &blocks[0]
        else {
            panic!("应为 Document 块: {blocks:?}")
        };
        assert_eq!(*source, DocSource::Url);
        assert_eq!(
            (media_type.as_str(), data.as_str()),
            ("application/pdf", uri)
        );
        assert_eq!(name.as_deref(), Some("s.pdf"));

        // 出站还原为 fileData part
        let contents = core_to_contents(&[Message {
            role: Role::User,
            content: vec![blocks[0].clone()],
            ext: Default::default(),
        }]);
        assert_eq!(
            contents[0]["parts"][0],
            json!({ "fileData": { "fileUri": uri, "mimeType": "application/pdf", "displayName": "s.pdf" } })
        );
    }

    /// 回归：executableCode/codeExecutionResult 入站不再丢弃——映射为
    /// ToolUse/ToolResult（EXEC_CODE 占位名，按序配对唯一 id）；回传 Gemini
    /// 还原为原生 part 形态。
    #[test]
    fn code_execution_parts_roundtrip() {
        let msgs = contents_to_core(&[json!({
            "role": "model",
            "parts": [
                { "executableCode": { "language": "PYTHON", "code": "print(1)" } },
                { "codeExecutionResult": { "outcome": "OUTCOME_OK", "output": "1\n" } },
                { "executableCode": { "language": "PYTHON", "code": "print(2)" } },
                { "codeExecutionResult": { "outcome": "OUTCOME_FAILED", "output": "err" } }
            ]
        })]);
        let content = &msgs[0].content;
        let (
            ContentBlock::ToolUse {
                id: i0,
                name: n0,
                input,
                ..
            },
            ContentBlock::ToolResult {
                tool_use_id: t0,
                is_error: e0,
                ..
            },
            ContentBlock::ToolUse { id: i1, .. },
            ContentBlock::ToolResult {
                tool_use_id: t1,
                is_error: e1,
                ..
            },
        ) = (&content[0], &content[1], &content[2], &content[3])
        else {
            panic!("应为 ToolUse/ToolResult 对: {content:?}")
        };
        assert_eq!(n0, EXEC_CODE);
        assert_eq!(input["code"], "print(1)");
        assert_eq!(t0, i0, "结果须回指配对调用");
        assert_eq!(t1, i1, "第二对结果须各自配对");
        assert_ne!(i0, i1, "id 须唯一");
        assert!(!e0 && *e1, "outcome 映射到 is_error");

        // 出站还原原生 part 形态
        let contents = core_to_contents(&msgs);
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(
            parts[0],
            json!({ "executableCode": { "language": "PYTHON", "code": "print(1)" } })
        );
        assert_eq!(
            parts[1],
            json!({ "codeExecutionResult": { "outcome": "OUTCOME_OK", "output": "1\n" } })
        );
        assert_eq!(
            parts[2],
            json!({ "executableCode": { "language": "PYTHON", "code": "print(2)" } })
        );
        assert_eq!(
            parts[3],
            json!({ "codeExecutionResult": { "outcome": "OUTCOME_FAILED", "output": "err" } })
        );
    }

    /// 回归：functionCall/functionResponse 无 id 时按名按序配对——同名函数
    /// 多次调用各得唯一 id，结果回指对应调用（此前结果以函数名为 id 与
    /// 调用侧 `{name}-call` 永不匹配，转 Anthropic 整请求 400）。
    /// 带真实 id 的部分原样保留。
    #[test]
    fn function_call_response_pair_by_name_in_order() {
        let msgs = contents_to_core(&[
            json!({
                "role": "model",
                "parts": [
                    { "functionCall": { "name": "read_file", "args": {"p": "a"} } },
                    { "functionCall": { "name": "read_file", "args": {"p": "b"} } },
                    { "functionCall": { "id": "real-id", "name": "ls", "args": {} } }
                ]
            }),
            json!({
                "role": "user",
                "parts": [
                    { "functionResponse": { "name": "read_file", "response": {"result": "A"} } },
                    { "functionResponse": { "name": "read_file", "response": {"result": "B"} } },
                    { "functionResponse": { "id": "real-id", "name": "ls", "response": {"result": "ok"} } }
                ]
            }),
        ]);

        let calls: Vec<&ContentBlock> = msgs[0]
            .content
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolUse { .. }))
            .collect();
        let id0 = match calls[0] {
            ContentBlock::ToolUse { id, .. } => id.clone(),
            _ => unreachable!(),
        };
        let id1 = match calls[1] {
            ContentBlock::ToolUse { id, .. } => id.clone(),
            _ => unreachable!(),
        };
        assert_ne!(id0, id1, "同名调用须分配唯一 id");

        let results: Vec<&ContentBlock> = msgs[1]
            .content
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolResult { .. }))
            .collect();
        let ContentBlock::ToolResult {
            tool_use_id: t0,
            content: c0,
            ..
        } = results[0]
        else {
            panic!()
        };
        let ContentBlock::ToolResult {
            tool_use_id: t1,
            content: c1,
            ..
        } = results[1]
        else {
            panic!()
        };
        let ContentBlock::ToolResult {
            tool_use_id: t2, ..
        } = results[2]
        else {
            panic!()
        };
        assert_eq!(t0, &id0, "第一个结果回指第一个同名调用");
        assert_eq!(t1, &id1, "第二个结果回指第二个同名调用");
        assert_eq!(t2, "real-id", "带真实 id 的结果原样保留");
        assert!(matches!(&c0[0], ContentBlock::Text { text } if text == "A"));
        assert!(matches!(&c1[0], ContentBlock::Text { text } if text == "B"));

        // 出站回传：id_to_name 解析回函数名，id 一并还原
        let contents = core_to_contents(&msgs);
        let frs: Vec<&Value> = contents[1]["parts"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p.get("functionResponse").is_some())
            .collect();
        assert_eq!(frs[0]["functionResponse"]["name"], "read_file");
        assert_eq!(frs[0]["functionResponse"]["id"], id0.as_str());
        assert_eq!(frs[2]["functionResponse"]["id"], "real-id");
    }

    /// 回归：is_error 经 response.is_error 保留位双向保真。
    #[test]
    fn tool_result_is_error_roundtrips() {
        let msgs = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::ToolUse {
                    id: "t1".into(),
                    item_id: None,
                    name: "exec".into(),
                    namespace: None,
                    input: json!({}),
                    signature: None,
                }],
                ext: Default::default(),
            },
            Message {
                role: Role::Tool,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "t1".into(),
                    content: vec![ContentBlock::text("boom")],
                    is_error: true,
                }],
                ext: Default::default(),
            },
        ];
        let contents = core_to_contents(&msgs);
        let fr = &contents[1]["parts"][0]["functionResponse"];
        assert_eq!(fr["response"]["is_error"], true);
        assert_eq!(fr["response"]["result"], "boom");

        let back = contents_to_core(&contents);
        let ContentBlock::ToolResult { is_error, .. } = &back[1].content[0] else {
            panic!()
        };
        assert!(*is_error, "is_error 回传应保留");
    }

    /// tool_result 内嵌文档随图片一并 hoist 为同回合 part。
    #[test]
    fn tool_result_documents_hoist_as_parts() {
        let msgs = vec![Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "f".into(),
                content: vec![
                    ContentBlock::text("see doc"),
                    ContentBlock::Document {
                        source: DocSource::Url,
                        media_type: "application/pdf".into(),
                        data: "https://x/d.pdf".into(),
                        name: None,
                    },
                    ContentBlock::Document {
                        source: DocSource::Base64,
                        media_type: "application/pdf".into(),
                        data: "PP".into(),
                        name: None,
                    },
                ],
                is_error: false,
            }],
            ext: Default::default(),
        }];
        let contents = core_to_contents(&msgs);
        // functionResponse 独占 contents[0]，标注与文档 part 拆入 contents[1]
        assert_eq!(contents.len(), 2);
        let fr_parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(fr_parts.len(), 1);
        assert_eq!(
            fr_parts[0]["functionResponse"]["response"]["result"],
            "see doc"
        );
        let parts = contents[1]["parts"].as_array().unwrap();
        assert_eq!(
            parts[0],
            json!({ "text": "[media content returned by function call f]" })
        );
        assert_eq!(
            parts[1],
            json!({ "fileData": { "fileUri": "https://x/d.pdf", "mimeType": "application/pdf" } })
        );
        assert_eq!(
            parts[2],
            json!({ "inlineData": { "mimeType": "application/pdf", "data": "PP" } })
        );
    }

    /// 纯文本 tool_result 不产出额外 part（保持既有结构）。
    #[test]
    fn text_only_tool_result_emits_no_extra_parts() {
        let msgs = vec![Message {
            role: Role::Tool,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "f".into(),
                content: vec![ContentBlock::text("ok")],
                is_error: false,
            }],
            ext: Default::default(),
        }];
        let contents = core_to_contents(&msgs);
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["functionResponse"]["response"]["result"], "ok");
    }

    #[test]
    fn maps_usage() {
        let u = usage_from_gemini(
            &json!({"promptTokenCount": 7, "candidatesTokenCount": 3, "cachedContentTokenCount": 2}),
        );
        assert_eq!(u.input_tokens, 7);
        assert_eq!(u.output_tokens, 3);
        assert_eq!(u.cache_read_tokens, 2);
    }

    /// Core 不变量：output 含 reasoning（thoughts）、input 含 toolUse prompt。
    /// 官方 totalTokenCount = prompt + toolUse + candidates + thoughts。
    #[test]
    fn usage_output_includes_thoughts_and_input_includes_tool_use() {
        let u = usage_from_gemini(&json!({
            "promptTokenCount": 7,
            "toolUsePromptTokenCount": 2,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 4,
        }));
        assert_eq!(u.input_tokens, 9);
        assert_eq!(u.output_tokens, 7);
        assert_eq!(u.reasoning_tokens, 4);

        // 出站 totalTokenCount 与官方口径一致（input + output 已各自含子项）
        let out = usage_object(&u);
        assert_eq!(out["totalTokenCount"], 16);
        assert_eq!(out["thoughtsTokenCount"], 4);
    }

    /// 收集 schema 位置中出现的全部对象键（用于递归断言违禁字段已被剥除）。
    /// `default`/`example`/`enum` 是实例数据而非 schema，不向下递归。
    fn collect_keys(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                for (k, val) in m {
                    out.push(k.clone());
                    if matches!(k.as_str(), "default" | "example" | "enum") {
                        continue;
                    }
                    collect_keys(val, out);
                }
            }
            Value::Array(arr) => arr.iter().for_each(|x| collect_keys(x, out)),
            _ => {}
        }
    }

    /// Kimi Code 等客户端的工具 schema 是完整 JSON Schema（draft-07/2020-12），
    /// `$schema`、`additionalProperties`、`propertyNames`、数值型 exclusive 边界
    /// 会被 Gemini 以 400 `Cannot find field` 整体拒绝，必须清洗为 OpenAPI 子集。
    #[test]
    fn tools_schema_sanitized_to_openapi_subset() {
        let tools = vec![Tool {
            name: "ask".into(),
            description: Some("d".into()),
            ext: Default::default(),
            input_schema: json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "timeout": {
                        "type": "integer",
                        "exclusiveMinimum": 0,
                        "description": "秒"
                    },
                    "ids": {
                        "type": "array",
                        "items": { "type": "string", "propertyNames": {} }
                    },
                    "choice": {
                        "anyOf": [
                            { "type": "string", "exclusiveMaximum": 5 },
                            { "type": "null" }
                        ]
                    },
                    "meta": {
                        "type": "object",
                        // default 是实例数据而非 schema，其中的键不得被清洗
                        "default": { "additionalProperties": true, "note": "keep" }
                    }
                },
                "required": ["timeout"]
            }),
        }];
        let out = core_to_tools(&tools).unwrap();
        let params = &out[0]["functionDeclarations"][0]["parameters"];

        let mut keys = Vec::new();
        collect_keys(&out, &mut keys);
        for banned in ["$schema", "additionalProperties", "propertyNames", "exclusiveMinimum", "exclusiveMaximum"] {
            assert!(!keys.iter().any(|k| k == banned), "违禁字段应被剥除: {banned}");
        }

        // 数值型 exclusive 边界放宽为闭区间
        assert_eq!(params["properties"]["timeout"]["minimum"], 0);
        assert_eq!(params["properties"]["choice"]["anyOf"][0]["maximum"], 5);
        // 单独 "null" 类型归一为 nullable（Gemini 的 type 是单值枚举）
        assert_eq!(params["properties"]["choice"]["anyOf"][1], json!({ "nullable": true }));
        // 嵌套 schema 位置递归清洗
        assert_eq!(params["properties"]["ids"]["items"]["type"], "string");
        // 支持字段与实例数据原样保留
        assert_eq!(params["type"], "object");
        assert_eq!(params["required"], json!(["timeout"]));
        assert_eq!(
            params["properties"]["meta"]["default"],
            json!({ "additionalProperties": true, "note": "keep" }),
            "default 中的实例数据不得按 schema 清洗"
        );
    }

    /// draft-07 布尔形态 exclusiveMinimum 本身丢弃，同行的 minimum 保留。
    #[test]
    fn boolean_exclusive_bound_dropped_but_minimum_kept() {
        let tools = vec![Tool {
            name: "t".into(),
            description: None,
            ext: Default::default(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "n": { "type": "integer", "minimum": 1, "exclusiveMinimum": true }
                }
            }),
        }];
        let out = core_to_tools(&tools).unwrap();
        let n = &out[0]["functionDeclarations"][0]["parameters"]["properties"]["n"];
        assert_eq!(n["minimum"], 1);
        assert!(n.get("exclusiveMinimum").is_none());
    }

    /// type 联合形态归一为首类型 + nullable；exclusive 边界与显式
    /// minimum/maximum 并存时取更严格者（与 key 书写顺序无关）。
    #[test]
    fn type_union_and_exclusive_bounds_normalized() {
        fn sanitize(s: Value) -> Value {
            sanitize_schema_logged(&s, &mut Vec::new())
        }
        assert_eq!(
            sanitize(json!({"type": ["string", "null"]})),
            json!({"type": "string", "nullable": true})
        );
        assert_eq!(sanitize(json!({"type": "null"})), json!({"nullable": true}));
        assert_eq!(
            sanitize(json!({"type": ["string", "integer"]})),
            json!({"type": "string"}),
            "多非 null 联合降级为首类型（Gemini 无法表达联合类型）"
        );
        // 合并取严：下界取大、上界取小，与书写顺序无关
        for s in [
            json!({"minimum": 0, "exclusiveMinimum": 5}),
            json!({"exclusiveMinimum": 5, "minimum": 0}),
        ] {
            assert_eq!(sanitize(s)["minimum"], 5);
        }
        for s in [
            json!({"maximum": 100, "exclusiveMaximum": 5}),
            json!({"exclusiveMaximum": 5, "maximum": 100}),
        ] {
            assert_eq!(sanitize(s)["maximum"], 5);
        }
    }

    /// 现行 GenAI 接口统一为 thinkingLevel 档位（thinkingBudget 已退出）；
    /// 与模型代际无关，只发 LOW/HIGH 公共子集。
    #[test]
    fn thinking_config_always_emits_level() {
        fn req_with_effort(model: &str, effort: &str) -> CoreRequest {
            let mut req = CoreRequest::new(model);
            req.max_tokens = Some(32000);
            req.reasoning = Some(moonbridge_core::Reasoning {
                effort: Some(effort.into()),
                summary: None,
            });
            req
        }

        let gc = core_to_generation_config(&req_with_effort("gemini-4-argon", "high")).unwrap();
        assert_eq!(gc["thinkingConfig"]["thinkingLevel"], "HIGH");
        assert!(gc["thinkingConfig"].get("thinkingBudget").is_none());

        // 档位映射与模型名无关：minimal/low → LOW，medium/high/xhigh/max → HIGH
        for (effort, level) in [("minimal", "LOW"), ("low", "LOW"), ("medium", "HIGH"), ("xhigh", "HIGH")] {
            let gc = core_to_generation_config(&req_with_effort("gemini-2.5-pro", effort)).unwrap();
            assert_eq!(gc["thinkingConfig"]["thinkingLevel"], level, "effort={effort}");
            assert!(gc["thinkingConfig"].get("thinkingBudget").is_none());
        }

        // none 意为关闭思考：不下发 thinkingConfig（绝不能反向放大为 HIGH）
        let gc = core_to_generation_config(&req_with_effort("gemini-4-argon", "none")).unwrap();
        assert!(gc.get("thinkingConfig").is_none());
        // 未知档位按 HIGH 下发（记 warn 日志）
        let gc = core_to_generation_config(&req_with_effort("gemini-4-argon", "ultra")).unwrap();
        assert_eq!(gc["thinkingConfig"]["thinkingLevel"], "HIGH");
    }
}
