---@meta
-- Moon Bridge Next 插件 API Stub（仅供 lua-language-server / LSP 使用，勿在插件中 require）。
--
-- 用法：在编辑器中把本文件加入 lua-language-server 的工作区库，即可获得 `mb.*` 与
-- `MB` 钩子的补全与类型提示。例如 VS Code（sumneko lua）settings.json：
--
--   "Lua.workspace.library": { "/path/to/moon-bridge-next/plugins": true }
--
-- 字段说明与 crates/plugin/src/{host,runtime,convert,bridge}.rs 保持同步。
--
-- 约定：插件脚本在全局暴露 `MB` table——既承载清单（name/version/scopes/capabilities），
-- 也承载钩子函数；宿主 API 挂载在全局 `mb`（小写）。沙箱中 `os/io/loadfile/dofile` 不可用。

--------------------------------------------------------------------------------
-- 清单与钩子（全局 MB 表）
--------------------------------------------------------------------------------

---@class MbVerdict
---报文钩子返回值：nil 表示放行（就地修改经 table 引用回写）；
---`{ action = "short_circuit", status?, headers?, body? }` 短路本次请求/响应；
---`{ action = "abort" }` 中止；
---`{ action = "retry", delay_ms? }` 延迟后重发本次上游请求（重走端点链）——仅
---`on_upstream_response_raw` 的**错误路径**（`msg.error == true`，上游非 2xx）消费，
---且是终局判定：立即生效、同阶段靠后的插件不再执行；其余阶段/成功路径拿到它一律
---按放行处理并继续扇出给后续插件；
---次数与延迟受网关配置 `pluginRetryMax`（默认 8，0 = 禁用插件重试）与
---`pluginRetryDelayCapMs`（默认 30000）钳制，超预算即放行、如实回传原始上游错误。
---@field action string|nil "short_circuit"|"abort"|"retry"|nil
---@field status number|nil short_circuit 时的 HTTP 状态码（默认 200）
---@field headers table[]|nil short_circuit 时的响应头 `{{k,v},...}`
---@field body any short_circuit 时的响应体（table/字符串）
---@field delay_ms number|nil retry 时的重试延迟（毫秒，缺省 0 = 立即重发）

---@class MbChunkVerdict
---chunk 钩子返回值：nil 放行；`{ action = "drop" }` 丢弃该 chunk。
---@field action string|nil "drop"|nil

---@class MbManifest
---@field name string 唯一插件名
---@field version string|nil 版本
---@field category string|nil 类别："core"（缺省，请求链路插件）|"quota"（配额查询插件：由配额引擎驱动，不进钩子注册表）
---@field scopes string[]|nil 作用域："global"|"provider"|"model"|"route"（仅 core 类生效）
---@field capabilities string[] 能力："core"|"raw_request"|"raw_response"|"raw_stream"（仅 core 类生效）
---@field requires table|nil 启用条件：`{ <网关配置键> = <期望值>, ... }`（bool/int/float/字符串）。
---仅 app 层校验——启用动作（新建即启用 / 停用→启用）时不满足则拒绝并弹提示；网关侧不处理。
---@field config_schema table|nil 配置 JSON Schema（供 UI 渲染表单）
---@field entry string|nil 入口提示
---@field init fun()|nil 加载后调用一次
---@field shutdown fun()|nil 卸载时调用
---Core IR 语义层钩子（需声明 "core"）。
---@field on_request (fun(ctx: MbCtx, req: MbCoreRequest): MbCoreRequest|nil)|nil
---@field on_response (fun(ctx: MbCtx, resp: MbCoreResponse): MbCoreResponse|nil)|nil
---@field inject_tools (fun(ctx: MbCtx): MbTool[])|nil
---@field filter_content (fun(ctx: MbCtx, block: table): boolean|table)|nil 返回 true 跳过该块
---@field on_stream_event (fun(ctx: MbCtx, ev: table): boolean)|nil 处理单个流事件；返回 true 丢弃该事件
---@field transform_error (fun(ctx: MbCtx, msg: string): string)|nil
---出入站原始报文钩子（需声明对应 raw_* 能力）。
---@field on_client_request_raw (fun(ctx: MbCtx, msg: MbRawMessage): MbVerdict|nil)|nil
---@field on_upstream_request_raw (fun(ctx: MbCtx, msg: MbRawMessage): MbVerdict|nil)|nil
---上游**非 2xx** 时也会触发本钩子（此时 msg.error == true，body 为错误原文文本）——此即
---错误路径：返回 `{ action = "retry", delay_ms = ? }` 让网关延迟后重发本轮请求
---（单端点 provider 的 5xx 自愈，见 plugins/utils/rescue_5xx.lua）。错误路径上 retry
---是终局判定（后续插件不再执行）；只处理错误路径的插件建议开头
---`if not msg.error then return end`，只处理成功的用 `if msg.error then return end`。
---流式请求另有第二种错误路径：上游返回 200 后**尚未产出任何内容**就以带内错误帧
---（如 responses 的 `response.failed`）宣告失败、或流未产出内容即静默终结——此时同样以
---`msg.error == true` 触发本钩子（`status == 200`，body 为上游错误文本，见
---plugins/utils/rescue_fake_200.lua）；零产出意味着客户端还什么都没收到，retry 是
---安全的。一旦上游产出过内容，流不可重试、本钩子不触发。
---错误路径上对报文的改写不回流客户端（客户端看到的错误消息由 core 层 transform_error 决定）。
---@field on_upstream_response_raw (fun(ctx: MbCtx, msg: MbRawMessage): MbVerdict|nil)|nil
---@field on_client_response_raw (fun(ctx: MbCtx, msg: MbRawMessage): MbVerdict|nil)|nil
---流式 chunk 钩子（需声明 "raw_stream"；高频，未声明则完全跳过）。
---@field on_upstream_chunk_raw (fun(ctx: MbCtx, chunk: MbRawChunk): MbChunkVerdict|nil)|nil
---@field on_client_chunk_raw (fun(ctx: MbCtx, chunk: MbRawChunk): MbChunkVerdict|nil)|nil
---配额查询（category = "quota" 插件）：由 Provider 绑定并调用 `MB.query`，
---不进钩子注册表、不参与上面的能力过滤与钩子链路（见 crates/gateway/src/quota.rs）。
---@field query (fun(ctx: MbQuotaQueryCtx): MbQuotaReturn)|nil 查询一次配额（仅 quota 类插件）

--------------------------------------------------------------------------------
-- 配额查询（MB.query，quota 类插件）
--------------------------------------------------------------------------------

---@class MbQuotaEntry
---@field type string 必填判别："percentage" 百分比额度 | "quota" 金额额度 | "counter" 计数器（前端不渲染）
---@field label string 配额展示名（如「5 小时窗口」）
---@field period_secs integer|nil 窗口秒数（如 5*3600），驱动前端本地消耗统计
---@field used_percent number|nil 已用百分比（percentage；与 left_percent 互补，只给一个即可）
---@field left_percent number|nil 剩余百分比（percentage）
---@field unit string|nil 金额/数量单位（如 "¥"、"GB"），quota/counter 使用
---@field used_amount number|nil 已用金额/数量（quota/counter；amount 与 percent 互不推导）
---@field left_amount number|nil 剩余金额/数量（quota）
---@field reset_at string|integer|nil 重置时间：字符串原样展示；或给 unix 秒数字（引擎归一为字符串，前端按本地时间格式化——沙箱无 os 库，毫秒时间戳请除 1000 取整后给出）

---@class MbQuotaReturn
---返回 table，经宿主序列化为 JSON 写入数据库。
---@field status string|nil "ok"|"error"（缺省 = ok）
---@field message string|nil 失败原因（status = "error" 时展示给用户）
---@field quotas MbQuotaEntry[]|nil 配额列表（缺 type 或缺该类型必填字段的项被剔除并记入 warnings）
---@field summary string|nil 一句话摘要
---额外字段原样保留在结果 payload 中。

---@class MbQuotaQueryCtx
---@field provider string Provider key（绑定该插件的上游服务）
---@field name string Provider key（同 provider，兼容别名）
---@field key string 本次查询的 API Key（与 keys[1] 相同）
---@field keys string[] 当前 key 的单元素数组——引擎对 Provider 的每个端点 key 各调用一次 MB.query 并拆行写入数据库，脚本只需按单 key 编写
---@field base_url string 该 key 所属端点的 base_url（配额接口基准地址；无端点 Provider 为空字符串）
---@field extra table Provider 配额配置（编辑表单「配额配置 JSON」的解码值）

---@type MbManifest
MB = {}

--------------------------------------------------------------------------------
-- 上下文（每次钩子调用注入的只读 table）
--------------------------------------------------------------------------------

---@class MbCtx
---@field request_id string 请求 ID
---@field session_id string|nil 会话 ID（无会话时为 nil）
---@field model_alias string 路由前模型别名
---@field client_protocol string 入口协议："openai-response"|"openai-chat"|"anthropic"|"google-genai"
---@field upstream_protocol string|nil 上游协议（未路由时为 nil）
---@field provider string|nil 上游服务 key（未路由时为 nil）
---@field stream boolean 是否流式请求

--------------------------------------------------------------------------------
-- Core IR（on_request / on_response 的操作对象；就地修改即生效）
--------------------------------------------------------------------------------

---@class MbTool
---@field name string
---@field description string|nil
---@field input_schema table|nil JSON Schema

---@class MbCoreRequest
---@field model string 路由解析后的上游模型名
---@field model_alias string 客户端请求的模型别名（路由前）
---@field system table system 指令（ContentBlock 数组，独立于 messages）
---@field messages table 消息数组
---@field tools MbTool[]
---@field tool_choice table|nil
---@field max_tokens number|nil
---@field temperature number|nil
---@field top_p number|nil
---@field stop string[]
---@field stream boolean
---@field reasoning table|nil 推理配置
---@field meta table|nil 请求级元数据（session_id、原始 headers、客户端标识等）

---@class MbCoreResponse
---@field content table ContentBlock 数组
---@field stop_reason string|nil "end_turn"|"max_tokens"|...
---@field usage table|nil { input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens }

--------------------------------------------------------------------------------
-- 原始报文（raw_* 钩子；body 为 JSON table / 字符串 / nil）
--------------------------------------------------------------------------------

---@class MbRawMessage
---@field stage string "client_request"|"upstream_request"|"upstream_response"|"client_response"
---@field protocol string
---@field provider string|nil
---@field method string
---@field url string
---@field status number
---@field headers table[] 请求/响应头，保序数组 `{{k,v},...}`（配 mb.headers 操作）
---@field body any JSON table / 文本字符串 / nil；超上限时为 nil 且 body_truncated=true
---@field body_truncated boolean|nil
---@field session_id string|nil 会话身份覆写通道：仅 client_request 阶段可写生效——
---    改成客户端自带会话 id（如 x-opencode-session 头值）即成为最高优先级身份源；
---    设 nil = 否决宿主已提取身份。其他阶段回读当前已解析值，改写无效果。
---@field error boolean|nil 上游错误路径标记：仅 upstream_response 阶段在上游非 2xx、
---    或流式请求零产出即被上游以带内错误帧宣告失败/静默终结时为 true（此时 status=200）
---    （只读，插件改写无效果）；据此区分「成功响应」与「错误路径」。

---@class MbRawChunk
---@field stage string "upstream_chunk"|"client_chunk"
---@field protocol string
---@field provider string|nil
---@field event string SSE event 名
---@field data any JSON table / 字符串 / nil；超上限时为 nil 且 data_truncated=true
---@field data_truncated boolean|nil
---@field raw string 原始报文文本

--------------------------------------------------------------------------------
-- 宿主 API（全局 mb 表；网络与跨 provider 调用为异步函数，可直接 await 语义使用）
--------------------------------------------------------------------------------

---@class MbHeaderList
---保序 header 数组，形如 `{{"Content-Type","application/json"}}`；元素为 {string, string}。

---@class MbHeadersHelper
---@field get fun(hs: MbHeaderList, name: string): string|nil 大小写不敏感读取
---@field set fun(hs: MbHeaderList, name: string, val: string): MbHeaderList 存在则改，否则追加
---@field remove fun(hs: MbHeaderList, name: string): MbHeaderList 移除同名项（全部）

---@class MbHttpRequest
---@field method string|nil 默认 "GET"
---@field url string 完整 URL（经宿主 egress proxy 与默认超时；**无目标白名单/过滤**——集中≠授权）
---@field headers MbHeaderList|nil
---@field body table|nil JSON 请求体
---@field timeout_ms number|nil

---@class MbHttpResponse
---@field status number
---@field headers MbHeaderList
---@field body any JSON 优先解析，否则为字符串

---@class MbHostApi
---日志（target 为 "plugin"，带插件名）。
---@field log { debug: fun(msg: string), info: fun(msg: string), warn: fun(msg: string), error: fun(msg: string) }
---插件配置：来自 store 中该插件的 config JSON。
---@field config table
---跨请求会话状态（按当前会话定位；key/value 均可序列化）。
---@field session { get: fun(key: string): any, set: fun(key: string, val: any) }
---受控 HTTP 子请求（async）。
---@field http { request: async fun(req: MbHttpRequest): MbHttpResponse }
---跨 provider 编排调用：以 Core IR 直接请求另一 provider（async）。
---@field provider { invoke: async fun(providerKey: string, model: string, req: MbCoreRequest): MbCoreResponse }
---摘要与编码工具。
---@field crypto { sha256: fun(s: string): string, hmac_sha256: fun(key: string, msg: string): string, base64_encode: fun(s: string): string, base64_decode: fun(s: string): string }
---header 辅助（大小写不敏感，操作保序数组）。
---@field headers MbHeadersHelper

---@type MbHostApi
mb = {}
