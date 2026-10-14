//! 网关引导配置（监听地址、认证、egress 代理、配额等）。
//!
//! 与 SQLite 中的业务配置分离：这些是「如何启动网关」的运行时参数，由 app 层
//! 从 `app_config_dir/config.toml` 加载后传入。

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewayConfig {
    /// 监听地址，如 `127.0.0.1:38440`。
    #[serde(default = "default_addr")]
    pub addr: String,
    /// Bearer 认证 token；`None` 表示不鉴权。
    #[serde(default)]
    pub auth_token: Option<String>,
    /// 所有上游调用的出站代理（HTTP(S)）。
    #[serde(default)]
    pub egress_proxy: Option<String>,
    /// body 大小上限（字节）。三处生效：入站请求体（axum DefaultBodyLimit）、
    /// 报文层钩子处理（超限降级为只读并跳过 Lua）、上游非流式/错误响应体读取
    /// （超限以 502 拒绝）。默认 100MB：大上下文/多图/长文档场景足够，
    /// 又保留了对失控请求的最后防线。
    #[serde(default = "default_max_body")]
    pub max_body_bytes: usize,
    /// 插件脚本目录：`script_ref` 里的相对 `.lua` 路径归一到此处，绝对路径也必须
    /// 落在其中才允许加载（见 `crate::read_script`）。`None` 时退化为按进程 CWD 解析。
    #[serde(default)]
    pub plugins_dir: Option<String>,
    /// 上游请求超时（秒）。
    ///
    /// 两处生效：非流式上游请求的总超时（`dispatch.rs` 发送 + 响应体读取共用
    /// 这一预算——它是整条请求的总量，端点故障转移与插件重试的各次发送/等待
    /// 都扣同一时钟，重试延迟超出剩余预算时直接放弃重试；流式刻意不受此限——
    /// 长生成不应被网关截断），以及插件沙箱的 `call_timeout` 派生基准
    /// （见 `lib.rs::sandbox_limits`）。
    /// 上游 HTTP 客户端本身只设连接超时，见 `upstream.rs::build_client`。
    #[serde(default = "default_timeout")]
    pub request_timeout_secs: u64,
    /// 流式上游相邻两个 SSE 事件之间的空闲上限（秒）。
    ///
    /// 只约束「无数据间隔」，不截断持续产出的长生成。中转把连接挂起但停止
    /// 发数据时，`chunks.next()` 会无限 pend——没有它整条流僵到客户端先断开，
    /// 审计只能留下「客户端断开或上游提前关闭」的假象；超时则按上游错误收尾：
    /// 向客户端发带内 error 事件 + 终结帧，usage/trace 如实记 `error`。
    /// `0` 表示禁用（恢复无限等待的旧行为）。
    #[serde(default = "default_stream_idle_timeout")]
    pub stream_idle_timeout_secs: u64,
    /// trace 文件目录；`None` 表示不写入磁盘。
    #[serde(default)]
    pub trace_dir: Option<String>,
    /// trace 是否记录请求/响应**体**；关闭时只留元数据（方法/URL/头/用量）。
    /// 体可能含用户对话内容与敏感业务数据，按需开关。
    #[serde(default = "default_trace_record_bodies")]
    pub trace_record_bodies: bool,
    /// trace 保留条数：写入磁盘后按 mtime 只保留最近 N 条，`0` 表示不清理。
    #[serde(default = "default_trace_retention")]
    pub trace_retention: usize,
    /// 会话水印开关：把 `[mb:xxxxxx]` 嵌进助手**推理块**明文首部，靠客户端下一轮带回
    /// 来识别会话。入站会先剥除干净再转发上游，上游模型永远看不到它。
    ///
    /// 面向的是**不携带任何会话标识**的客户端（如 Qwen Code：其 `prompt_cache_key`
    /// 注入以直连 `api.openai.com` 为前提）。若你的客户端已通过 `session_id` 字段、
    /// `previous_response_id` 或 `X-Codex-Window-Id` 头表明身份，可关掉本开关。
    /// 代价：标记出现在客户端本地 transcript 的 thinking 明文里（用户可见正文
    /// 不受污染），每轮占约十余 token。关掉后**仍会剥除**入站 marker（客户端
    /// 可能带着开启期间留下的历史），只是不再嵌入。
    #[serde(default = "default_session_marker")]
    pub session_marker: bool,
    /// 活跃会话表的深度上限（LRU，超深淘汰最久未命中者并连带清理其插件态）。
    /// 注意：uuid 载荷自带身份，淘汰不会改变 marker 会话的 session id——表只
    /// 承担外部身份 tag 映射与插件态回收。
    #[serde(default = "default_session_depth")]
    pub session_table_depth: usize,
    /// 插件驱动的上游重试上限（`config.toml` 键 `pluginRetryMax`）。
    ///
    /// 语义：上游响应为错误（非 2xx）且报文层钩子返回 `{action="retry"}` 时，
    /// 网关延迟后**重走整条端点链**（重新构造请求 → 出站钩子 → 发送）；本值是
    /// 单次请求允许的重试次数，`0` 表示禁用插件重试。这是插件的硬防护——
    /// 插件返回多少次 retry 都不会死循环（超出后按放行处理，如实回传上游错误）。
    /// 非流式请求的每次重试还受 `request_timeout_secs` 总预算约束：剩余预算
    /// 装不下重试延迟即放弃，把真实上游错误还给客户端。
    #[serde(default = "default_plugin_retry_max")]
    pub plugin_retry_max: usize,
    /// 插件重试的单次延迟上限（毫秒，`config.toml` 键 `pluginRetryDelayCapMs`）。
    /// 插件给的 `delay_ms` 会被钳到该值内；钩子阻塞的是本次请求的任务，
    /// 设得过大同样会拖长上游错误暴露给客户端的时间。非流式请求下钳制后的
    /// 延迟若仍超过 `request_timeout_secs` 剩余预算，该次重试同样被放弃。
    #[serde(default = "default_plugin_retry_delay_cap")]
    pub plugin_retry_delay_cap_ms: u64,
}

fn default_addr() -> String {
    "127.0.0.1:38440".to_string()
}
fn default_max_body() -> usize {
    100 * 1024 * 1024
}
fn default_timeout() -> u64 {
    300
}
fn default_stream_idle_timeout() -> u64 {
    120
}
fn default_session_marker() -> bool {
    true
}
fn default_trace_record_bodies() -> bool {
    true
}
fn default_trace_retention() -> usize {
    500
}
fn default_session_depth() -> usize {
    64
}
fn default_plugin_retry_max() -> usize {
    8
}
fn default_plugin_retry_delay_cap() -> u64 {
    30_000
}

impl Default for GatewayConfig {
    fn default() -> Self {
        GatewayConfig {
            addr: default_addr(),
            auth_token: None,
            egress_proxy: None,
            max_body_bytes: default_max_body(),
            plugins_dir: None,
            request_timeout_secs: default_timeout(),
            stream_idle_timeout_secs: default_stream_idle_timeout(),
            trace_dir: None,
            trace_record_bodies: default_trace_record_bodies(),
            trace_retention: default_trace_retention(),
            session_marker: default_session_marker(),
            session_table_depth: default_session_depth(),
            plugin_retry_max: default_plugin_retry_max(),
            plugin_retry_delay_cap_ms: default_plugin_retry_delay_cap(),
        }
    }
}
