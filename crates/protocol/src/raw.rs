//! 报文层数据载体：`RawMessage` / `RawChunk` 及其判定结果。
//!
//! 这些类型表示「协议转换最外层」的原始 HTTP 报文，供 Lua 插件的报文层钩子
//! 直接读写 headers / body / SSE chunk。Rust 内部用 tagged enum 保证 serde
//! 无歧义；plugin crate 负责把它们转换为 Lua 友好的 table 形态。

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 原始报文体。以 `kind` 判别，避免 untagged 在 JSON array 与 binary 间的歧义。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RawBody {
    Json {
        value: Value,
    },
    Text {
        text: String,
    },
    /// 二进制体（base64 传输，Lua 侧以字符串呈现）。
    Binary {
        data: Vec<u8>,
    },
    Empty,
}

impl RawBody {
    pub fn json(value: Value) -> Self {
        RawBody::Json { value }
    }
    pub fn text(text: impl Into<String>) -> Self {
        RawBody::Text { text: text.into() }
    }
    pub fn as_json(&self) -> Option<&Value> {
        match self {
            RawBody::Json { value } => Some(value),
            _ => None,
        }
    }
    pub fn as_json_mut(&mut self) -> Option<&mut Value> {
        match self {
            RawBody::Json { value } => Some(value),
            _ => None,
        }
    }
    pub fn as_text(&self) -> Option<&str> {
        match self {
            RawBody::Text { text } => Some(text),
            _ => None,
        }
    }
    /// 估算体积（字节），用于沙箱配额判断。
    pub fn approx_len(&self) -> usize {
        match self {
            RawBody::Json { value } => value.to_string().len(),
            RawBody::Text { text } => text.len(),
            RawBody::Binary { data } => data.len(),
            RawBody::Empty => 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawStage {
    ClientRequest,
    UpstreamRequest,
    UpstreamResponse,
    ClientResponse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkStage {
    UpstreamChunk,
    ClientChunk,
}

/// 完整 HTTP 报文（非流式四个阶段共用同一载体）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawMessage {
    pub stage: RawStage,
    pub protocol: moonbridge_core::Protocol,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// 保序 header 列表，可读写。
    #[serde(default)]
    pub headers: Vec<(String, String)>,
    #[serde(default = "default_body")]
    pub body: RawBody,
    /// 会话身份覆写通道：仅 `client_request` 阶段生效——钩子把它改成客户端
    /// 自带的会话 id（如 `x-opencode-session` 头值）即成为最高优先级身份源，
    /// 写入 `ctx.session_id` 后走外部身份分支；设 nil = 否决宿主已提取的身份。
    /// 其他阶段仅回读当前已解析值（改写无效果）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// 上游错误路径标记：仅 upstream_response 阶段在上游非 2xx 时为 true（只读，插件改写无效果）。
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub error: bool,
}

fn default_body() -> RawBody {
    RawBody::Empty
}

impl RawMessage {
    /// 大小写不敏感读取首个匹配 header。
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    /// 同名 header 存在时覆盖首个匹配，否则追加。
    pub fn set_header(&mut self, name: &str, value: impl Into<String>) {
        let value = value.into();
        if let Some(slot) = self
            .headers
            .iter_mut()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
        {
            slot.1 = value;
        } else {
            self.headers.push((name.to_string(), value));
        }
    }
    /// 移除所有匹配 header。
    pub fn remove_header(&mut self, name: &str) {
        self.headers.retain(|(k, _)| !k.eq_ignore_ascii_case(name));
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawChunk {
    pub stage: ChunkStage,
    pub protocol: moonbridge_core::Protocol,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// SSE `event:` 字段。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    /// SSE `data:` 字段解析结果（通常为 JSON）。
    #[serde(default = "default_body")]
    pub data: RawBody,
    /// 原始 `data:` 行文本（当 data 非 JSON 或需保真时使用）。
    #[serde(default)]
    pub raw: String,
}

impl RawChunk {
    pub fn json(
        stage: ChunkStage,
        protocol: moonbridge_core::Protocol,
        event: Option<String>,
        value: Value,
    ) -> Self {
        let raw = value.to_string();
        RawChunk {
            stage,
            protocol,
            provider: None,
            event,
            data: RawBody::json(value),
            raw,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum RawVerdict {
    /// 放行（使用就地修改后的报文继续）。
    Pass,
    /// 短路：网关直接用给定响应应答客户端，跳过上游。
    ShortCircuit {
        status: u16,
        #[serde(default)]
        headers: Vec<(String, String)>,
        #[serde(default = "default_body")]
        body: RawBody,
    },
    /// 重试当前上游请求：重走端点链（`from_core_request` → 出站钩子 → 发送），
    /// `delay_ms` 毫秒后发起。仅上游响应钩子在**错误路径**（`msg.error == true`）上
    /// 处理该动作，且是终局判定——立即生效，同阶段靠后的插件不再执行；其余阶段/
    /// 成功路径拿到它一律按放行处理、继续扇出给后续插件。错误路径有两类：
    /// 上游非 2xx；或流式请求 200 后零产出即以带内错误帧宣告失败（status=200、
    /// error=true）——后者客户端未收到任何内容，重发安全。
    /// 网关侧有硬上限与延迟钳制（`plugin_retry_max` / `plugin_retry_delay_cap_ms`），
    /// 插件返回多少次都不会死循环。
    Retry {
        #[serde(default)]
        delay_ms: u64,
    },
    /// 中止：返回错误给客户端。
    Abort { message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChunkVerdict {
    /// 转发（使用就地修改后的 chunk）。
    Forward,
    Drop,
}
