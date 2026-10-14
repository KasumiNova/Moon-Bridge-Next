//! 配额查询引擎与定时调度。
//!
//! 每个绑定配额查询的 Provider 指定一个 `category = "quota"` 的插件：宿主加载插件脚本
//! （复用 [`moonbridge_plugin::LuaRuntime`] 的沙箱与 `mb.*` 宿主 API），再调用
//! `MB.query(ctx)`，把返回值经 serde 转为 JSON 写入数据库。配额插件不进网关的
//! [`moonbridge_protocol::PluginHooks`] 注册表，宿主桥也只实现 `mb.http.request`——
//! 配额查询是旁路能力，不该与网关生命周期互相牵制。
//!
//! key 来源唯一：Provider 端点按序展开（`api_key` 为空的端点沿用上一个非空 key，与
//! 网关转发语义一致），引擎对每个端点各跑一次脚本（ctx 携带该端点的 `base_url` 与
//! 生效 key），结果按 `(provider_key, key_index)` 拆行写入数据库，`key_index` 即端点下标。
//! 无端点的 Provider 退化为单次 `key = ""` 查询（账号级接口）。写入数据库的 `key_label`
//! 是掩码后的展示标签，key 原文不进结果表。
//!
//! 失败分层（按端点独立）：
//! - **引擎级失败**（插件缺失 / 非 quota 类 / 脚本读不到 / 无 `MB.query` / 抛错 / 超时）→
//!   该行 `status = "error"`，`error` 记消息，`payload` **保留上一次**的值
//!   （前端仍能看到最后已知配额）；
//! - **脚本业务失败**（返回 table 且 `status = "error"`）→ 同样 `status = "error"`，
//!   但 `payload` 更新为该次返回——脚本常在失败时也带回诊断上下文。

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use moonbridge_core::{CoreRequest, CoreResponse};
use moonbridge_plugin::{
    HostBridge, HttpRequest, HttpResponse, LuaRuntime, SandboxLimits, SessionStore,
};
use moonbridge_store::{
    Database, PluginRecord, Provider, ProviderQuotaView, QuotaEntry, QuotaKeyResult, QuotaResult,
};
use serde_json::{json, Value};

use crate::parse_script_ref;
pub use crate::quota_http::QuotaNetworkPolicy;
use crate::ScriptRef;

/// 单次查询的整体超时（含脚本加载 + `MB.query` 执行）。
const QUERY_TIMEOUT: Duration = Duration::from_secs(45);

/// 调度器扫描周期：每轮挑出到期 Provider 串行执行。
pub const SCHEDULE_TICK: Duration = Duration::from_secs(30);

/// 当前 unix 时间戳（秒）。口径与 store 的 `created_at` 一致。
fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// 配额脚本的宿主桥：只提供受网络策略约束的 `mb.http.request`。
struct QuotaBridge {
    network_policy: QuotaNetworkPolicy,
    base_url: String,
    network_error: Arc<Mutex<Option<String>>>,
}

#[async_trait]
impl HostBridge for QuotaBridge {
    async fn http_request(&self, req: HttpRequest) -> Result<HttpResponse, String> {
        let result = self.network_policy.request(&self.base_url, req).await;
        if let Err(error) = &result {
            if let Ok(mut network_error) = self.network_error.lock() {
                *network_error = Some(error.clone());
            }
        }
        result
    }

    async fn provider_invoke(
        &self,
        _provider: &str,
        _model: &str,
        _req: CoreRequest,
    ) -> Result<CoreResponse, String> {
        Err("配额脚本不支持".to_string())
    }
}

struct KeyTarget {
    key: String,
    base_url: String,
}

#[derive(Clone)]
pub struct QuotaEngine {
    db: Arc<Database>,
    /// 脚本根目录：插件 `script_ref` 里相对 `.lua` 归一到此处，绝对路径必须落在其内。
    /// `None` 时不做包含性校验（CLI/测试场景，与插件加载侧同语义）。
    plugins_dir: Option<PathBuf>,
    network_policy: QuotaNetworkPolicy,
}

impl QuotaEngine {
    pub fn new(db: Arc<Database>, plugins_dir: Option<PathBuf>) -> Self {
        QuotaEngine {
            db,
            plugins_dir,
            network_policy: QuotaNetworkPolicy::from_environment(None),
        }
    }

    pub fn with_network_policy(mut self, network_policy: QuotaNetworkPolicy) -> Self {
        self.network_policy = network_policy;
        self
    }

    /// 覆写脚本根目录（宿主在启动时固定为与插件同一套 `plugins_dir`；测试也可用）。
    pub fn with_plugins_dir(mut self, plugins_dir: Option<PathBuf>) -> Self {
        self.plugins_dir = plugins_dir;
        self
    }

    /// 解析 Provider 绑定的配额插件：必须存在且 `category = "quota"`。
    fn resolve_plugin(&self, provider: &Provider) -> Result<PluginRecord, String> {
        let plugin_ref = provider.quota_plugin_ref.trim();
        if plugin_ref.is_empty() {
            return Err("未绑定配额查询插件".to_string());
        }
        let plugin = self
            .db
            .get_plugin(plugin_ref)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("配额插件 {plugin_ref} 不存在"))?;
        if plugin.category != "quota" {
            return Err(format!("插件 {plugin_ref} 不是配额查询类插件"));
        }
        Ok(plugin)
    }

    /// 端点按序展开为查询目标：`api_key` 为空的端点沿用上一个非空 key。
    /// 无端点时退化为单个空 key 目标（账号级查询接口）。
    fn key_targets(provider: &Provider) -> Vec<KeyTarget> {
        let mut targets = Vec::with_capacity(provider.endpoints.len().max(1));
        let mut carry = String::new();
        for ep in &provider.endpoints {
            if !ep.api_key.is_empty() {
                carry = ep.api_key.clone();
            }
            targets.push(KeyTarget {
                key: carry.clone(),
                base_url: ep.base_url.clone(),
            });
        }
        if targets.is_empty() {
            targets.push(KeyTarget {
                key: String::new(),
                base_url: String::new(),
            });
        }
        targets
    }

    /// 执行一个 Provider 的配额查询：逐端点各跑一次脚本、各自写入数据库，返回本次全部结果。
    ///
    /// 插件解析失败时落一行 `key_index = 0` 的错误结果并删除其余残留行——
    /// 前端呈现为单条错误行。
    pub async fn run_provider(
        &self,
        provider: &Provider,
    ) -> moonbridge_store::Result<Vec<QuotaKeyResult>> {
        let now = now_unix();
        let plugin = match self.resolve_plugin(provider) {
            Ok(p) => p,
            Err(message) => {
                let kr = QuotaKeyResult {
                    key_index: 0,
                    key_label: String::new(),
                    result: self.engine_error_result(&provider.key, 0, message, now)?,
                };
                self.db.upsert_quota_key_result(&provider.key, &kr)?;
                self.db.prune_quota_results(&provider.key, 1)?;
                return Ok(vec![kr]);
            }
        };
        let targets = Self::key_targets(provider);
        let mut out = Vec::with_capacity(targets.len());
        for (idx, target) in targets.iter().enumerate() {
            out.push(
                self.run_key(provider, &plugin, idx as i64, target, now)
                    .await?,
            );
        }
        self.db
            .prune_quota_results(&provider.key, targets.len() as i64)?;
        Ok(out)
    }

    /// 刷新一个 Provider 并回传视图（供 REST/IPC）。数据库失败始终上抛。
    pub async fn refresh_provider(
        &self,
        provider: &Provider,
    ) -> moonbridge_store::Result<ProviderQuotaView> {
        self.run_provider(provider).await?;
        Ok(
            view_of(&self.db, &provider.key)?.unwrap_or_else(|| ProviderQuotaView {
                provider_key: provider.key.clone(),
                quota_plugin_ref: provider.quota_plugin_ref.clone(),
                quota_interval_secs: provider.quota_interval_secs,
                quota_enabled: provider.quota_enabled,
                key_count: provider.endpoints.len() as i64,
                results: Vec::new(),
            }),
        )
    }

    /// 试运行一个 Provider 的配额查询：不写入数据库、不读历史结果，所有失败仅返回预览。
    pub async fn test_provider(&self, provider: &Provider) -> Vec<QuotaKeyResult> {
        let now = now_unix();
        let plugin = match self.resolve_plugin(provider) {
            Ok(p) => p,
            Err(message) => {
                return vec![QuotaKeyResult {
                    key_index: 0,
                    key_label: String::new(),
                    result: QuotaResult {
                        status: "error".to_string(),
                        payload: None,
                        error: Some(message),
                        queried_at: now,
                    },
                }];
            }
        };
        let targets = Self::key_targets(provider);
        let mut out = Vec::with_capacity(targets.len());
        for (idx, target) in targets.iter().enumerate() {
            let ctx = Self::ctx_for(provider, target);
            let result = match self.query(provider, &plugin, target, &ctx).await {
                Ok(ret) => Self::from_script_return(&ret, now),
                Err(message) => QuotaResult {
                    status: "error".to_string(),
                    payload: None,
                    error: Some(message),
                    queried_at: now,
                },
            };
            out.push(QuotaKeyResult {
                key_index: idx as i64,
                key_label: mask_key(&target.key),
                result,
            });
        }
        out
    }

    /// 串行刷新全部绑定且启用的 Provider，返回全部配额视图；数据库失败中断并上抛。
    pub async fn refresh_all(&self) -> moonbridge_store::Result<Vec<ProviderQuotaView>> {
        for provider in self
            .db
            .list_providers()?
            .into_iter()
            .filter(|p| !p.quota_plugin_ref.trim().is_empty() && p.quota_enabled)
        {
            self.run_provider(&provider).await?;
        }
        list_views(&self.db)
    }

    async fn run_key(
        &self,
        provider: &Provider,
        plugin: &PluginRecord,
        key_index: i64,
        target: &KeyTarget,
        now: i64,
    ) -> moonbridge_store::Result<QuotaKeyResult> {
        let ctx = Self::ctx_for(provider, target);
        let result = match self.query(provider, plugin, target, &ctx).await {
            Ok(ret) => Self::from_script_return(&ret, now),
            Err(message) => self.engine_error_result(&provider.key, key_index, message, now)?,
        };
        let kr = QuotaKeyResult {
            key_index,
            key_label: mask_key(&target.key),
            result,
        };
        self.db.upsert_quota_key_result(&provider.key, &kr)?;
        Ok(kr)
    }

    /// 引擎级失败的结果：`payload` 保留该行上一次写入数据库的值。
    fn engine_error_result(
        &self,
        provider_key: &str,
        key_index: i64,
        message: String,
        now: i64,
    ) -> moonbridge_store::Result<QuotaResult> {
        let previous = self.db.get_quota_key_result(provider_key, key_index)?;
        Ok(QuotaResult {
            status: "error".to_string(),
            payload: previous.and_then(|p| p.result.payload),
            error: Some(message),
            queried_at: now,
        })
    }

    /// 读取插件脚本源码：经 [`parse_script_ref`] 收敛到 `plugins_dir` 内。
    fn read_script(&self, script_ref: &str) -> Result<String, String> {
        match parse_script_ref(script_ref, self.plugins_dir.as_deref()) {
            ScriptRef::Inline(src) => Ok(src),
            ScriptRef::Rejected(p) => Err(format!("脚本路径越出插件目录: {p}")),
            ScriptRef::File(path) => std::fs::read_to_string(&path)
                .map_err(|e| format!("读取脚本 {} 失败: {e}", path.display())),
        }
    }

    /// 构造单端点的脚本 ctx：引擎对每个端点各跑一次脚本，脚本只需按单 key 编写
    /// （`key` 与 `keys` 同值，`keys` 恒为单元素数组）。`base_url` 取该端点的
    /// 转发地址；`extra` 为 Provider 上的配额配置（`quota_config`）。
    fn ctx_for(provider: &Provider, target: &KeyTarget) -> Value {
        json!({
            "name": provider.key,
            "key": target.key,
            "keys": [target.key],
            "base_url": target.base_url,
            "provider": provider.key,
            "extra": provider.quota_config,
        })
    }

    /// 一次性执行脚本并返回 `MB.query` 的返回值（JSON）。
    async fn query(
        &self,
        provider: &Provider,
        plugin: &PluginRecord,
        target: &KeyTarget,
        ctx: &Value,
    ) -> Result<Value, String> {
        let deadline = tokio::time::Instant::now() + QUERY_TIMEOUT;
        let query = async {
            let engine = self.clone();
            let script_ref = plugin.script_ref.clone();
            let script = tokio::task::spawn_blocking(move || engine.read_script(&script_ref))
                .await
                .map_err(|_| "读取配额脚本失败".to_string())??;
            let network_error = Arc::new(Mutex::new(None));
            let bridge = Arc::new(QuotaBridge {
                network_policy: self.network_policy.clone(),
                base_url: target.base_url.clone(),
                network_error: network_error.clone(),
            });
            let limits = SandboxLimits {
                call_timeout: deadline.saturating_duration_since(tokio::time::Instant::now()),
                ..SandboxLimits::default()
            };
            let runtime = LuaRuntime::new_with_limits(
                &provider.key,
                &script,
                &provider.quota_config,
                bridge,
                SessionStore::new(),
                limits,
            )
            .map_err(|_| "配额脚本加载失败".to_string())?;
            if tokio::time::Instant::now() >= deadline {
                return Err("配额脚本查询超时".to_string());
            }
            runtime.call_mb_once("query", ctx).await.map_err(|_| {
                network_error
                    .lock()
                    .ok()
                    .and_then(|mut error| error.take())
                    .unwrap_or_else(|| "配额脚本执行失败".to_string())
            })
        };
        tokio::time::timeout_at(deadline, query)
            .await
            .map_err(|_| format!("查询超时（{} 秒）", QUERY_TIMEOUT.as_secs()))?
    }

    /// 只有对象中的显式 `status = "ok"` 视为成功；业务错误更新 payload。
    /// `quotas` 按 [`QuotaEntry`] 归一化，空数组也是有效结果。
    fn from_script_return(ret: &Value, now: i64) -> QuotaResult {
        let valid_status = ret
            .as_object()
            .and_then(|obj| obj.get("status"))
            .and_then(Value::as_str);
        let status = if valid_status == Some("ok") {
            "ok"
        } else {
            "error"
        };
        let error = match valid_status {
            Some("ok") => None,
            Some("error") => Some(
                ret.get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("脚本返回错误状态")
                    .to_string(),
            ),
            _ => Some("配额脚本必须返回包含有效 status（ok/error）的对象".to_string()),
        };
        let payload = ret.is_object().then(|| normalize_quotas(ret));
        QuotaResult {
            status: status.to_string(),
            payload,
            error,
            queried_at: now,
        }
    }
}

/// key 的展示掩码：空串 → `""`；长度 ≤ 4 → `****`；≤ 10 → 前 2 位 + `…`；
/// 更长 → 前 6 位 + `…` + 后 4 位。结果的 key_label 只保存掩码，不保存完整 key。
fn mask_key(key: &str) -> String {
    let chars: Vec<char> = key.chars().collect();
    let len = chars.len();
    if len == 0 {
        return String::new();
    }
    if len <= 4 {
        return "****".to_string();
    }
    if len <= 10 {
        let prefix: String = chars[..2].iter().collect();
        return format!("{prefix}…");
    }
    let prefix: String = chars[..6].iter().collect();
    let suffix: String = chars[len - 4..].iter().collect();
    format!("{prefix}…{suffix}")
}

/// 归一化 `payload.quotas`：逐元素解析为 [`QuotaEntry`]
/// （snake_case → camelCase），补齐 used/left percent 互补值。
///
/// 严格契约：`type` 必填且只认 `percentage|quota|counter`；`percentage` 至少有一个
/// percent 字段，`quota`/`counter` 至少有一个 amount 字段。非法元素被剔除并把原因记进
/// `payload.warnings`（追加到脚本已有的 warnings 后），不静默丢弃也不使整份 payload 失败。
/// `quotas` 缺失或不是数组时整份 payload 原样返回（脚本诊断字段不干预）。
fn normalize_quotas(ret: &Value) -> Value {
    let Some(quotas) = ret.get("quotas") else {
        return ret.clone();
    };
    // Lua 空表序列化为 {} 而非 []，归一成空数组
    if quotas.as_object().is_some_and(|o| o.is_empty()) {
        let mut out = ret.clone();
        out["quotas"] = json!([]);
        return out;
    }
    let Some(items) = quotas.as_array() else {
        return ret.clone();
    };
    let mut parsed: Vec<QuotaEntry> = Vec::with_capacity(items.len());
    let mut warnings: Vec<String> = ret
        .get("warnings")
        .and_then(Value::as_array)
        .map(|w| {
            w.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    for (i, item) in items.iter().enumerate() {
        let label = item
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let where_ = || format!("配额 #{i}（{label}）");
        let mut q = match serde_json::from_value::<QuotaEntry>(item.clone()) {
            Ok(q) => q,
            Err(_) => {
                warnings.push(format!("{}字段形状非法", where_()));
                continue;
            }
        };
        let missing = match q.quota_type.as_str() {
            "percentage" => {
                match (q.used_percent, q.left_percent) {
                    (Some(u), None) => q.left_percent = Some((100.0 - u).clamp(0.0, 100.0)),
                    (None, Some(l)) => q.used_percent = Some((100.0 - l).clamp(0.0, 100.0)),
                    _ => {}
                }
                if q.used_percent.is_none() && q.left_percent.is_none() {
                    Some("percent 字段")
                } else {
                    None
                }
            }
            "quota" => (q.used_amount.is_none() && q.left_amount.is_none())
                .then_some("used_amount/left_amount 字段"),
            "counter" => q.used_amount.is_none().then_some("used_amount 字段"),
            other => {
                warnings.push(format!("{}type 无效: {other}", where_()));
                continue;
            }
        };
        if let Some(missing) = missing {
            warnings.push(format!("{}缺少{missing}", where_()));
            continue;
        }
        parsed.push(q);
    }
    let mut out = ret.clone();
    if let Some(obj) = out.as_object_mut() {
        obj.insert("quotas".to_string(), json!(parsed));
        if !warnings.is_empty() {
            obj.insert("warnings".to_string(), json!(warnings));
        }
    }
    out
}

/// 内置配额插件种子：`plugins/quota/` 下的官方适配器，首次启动时以**内联
/// `script_ref`** 落进 `plugins` 表（`category = "quota"`、默认启用），供 Provider
/// 直接绑定。`config_schema` 由沙箱求值 `MB` 清单填入，驱动 Provider 编辑页的
/// 配额配置表单。用户可像普通插件一样改脚本/停用/删除——已播过种的库不再重播
/// （settings 标记 [`moonbridge_store::QUOTA_SEEDS_DONE`]），用户删掉的内置插件不会复活。
pub const BUILTIN_QUOTA_PLUGINS: &[(&str, &str)] = &[
    (
        "new-api",
        include_str!("../../../plugins/quota/new-api.lua"),
    ),
    (
        "sub2api",
        include_str!("../../../plugins/quota/sub2api.lua"),
    ),
    (
        "deepseek",
        include_str!("../../../plugins/quota/deepseek.lua"),
    ),
    (
        "moonshot",
        include_str!("../../../plugins/quota/moonshot.lua"),
    ),
    (
        "siliconflow",
        include_str!("../../../plugins/quota/siliconflow.lua"),
    ),
    (
        "openrouter",
        include_str!("../../../plugins/quota/openrouter.lua"),
    ),
    (
        "zhipu-glm",
        include_str!("../../../plugins/quota/zhipu-glm.lua"),
    ),
    (
        "kimi-coding",
        include_str!("../../../plugins/quota/kimi-coding.lua"),
    ),
    (
        "commandcode",
        include_str!("../../../plugins/quota/commandcode.lua"),
    ),
    (
        "claude-code",
        include_str!("../../../plugins/quota/claude-code.lua"),
    ),
    (
        "generic-percent",
        include_str!("../../../plugins/quota/generic-percent.lua"),
    ),
    (
        "generic-amount",
        include_str!("../../../plugins/quota/generic-amount.lua"),
    ),
];

/// 首次启动播种内置配额插件；已播过种的库直接跳过（返回 Ok(false)）。
/// 同名插件已存在时跳过该项，不覆盖用户同名记录。
pub fn seed_builtin_quota_plugins(db: &Database) -> moonbridge_store::Result<bool> {
    if db
        .get_setting(moonbridge_store::QUOTA_SEEDS_DONE)?
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return Ok(false);
    }
    for (name, script) in BUILTIN_QUOTA_PLUGINS {
        if db.get_plugin(name)?.is_some() {
            continue;
        }
        let manifest = LuaRuntime::manifest_of_script(name, script).ok();
        db.upsert_plugin(&moonbridge_store::PluginRecord {
            name: (*name).to_string(),
            source: "lua".to_string(),
            script_ref: (*script).to_string(),
            enabled: true,
            config: serde_json::Value::Null,
            scopes: Vec::new(),
            capabilities: Vec::new(),
            category: "quota".to_string(),
            config_schema: manifest
                .and_then(|m| m.config_schema)
                .unwrap_or(serde_json::Value::Null),
        })?;
    }
    db.set_setting(moonbridge_store::QUOTA_SEEDS_DONE, &serde_json::json!(true))?;
    Ok(true)
}

/// 列出全部绑定配额查询的 Provider 视图（附逐端点的最近一次结果）。
///
/// REST 与 IPC 两个宿主共用，保证 `list` / `refresh_all` 的返回形状一致。
pub fn list_views(db: &Database) -> moonbridge_store::Result<Vec<ProviderQuotaView>> {
    db.list_quota_views()
}

/// 取单个 Provider 的配额视图（Provider 不存在时 `Ok(None)`）。
pub fn view_of(
    db: &Database,
    provider_key: &str,
) -> moonbridge_store::Result<Option<ProviderQuotaView>> {
    let Some(provider) = db.get_provider(provider_key)? else {
        return Ok(None);
    };
    let results = db.list_quota_results(provider_key)?;
    Ok(Some(ProviderQuotaView {
        provider_key: provider.key.clone(),
        quota_plugin_ref: provider.quota_plugin_ref.clone(),
        quota_interval_secs: provider.quota_interval_secs,
        quota_enabled: provider.quota_enabled,
        key_count: provider.endpoints.len() as i64,
        results,
    }))
}

/// 启动配额定时调度（常驻任务：应用存活期间一直跑，不提供停止句柄）。
///
/// 每 [`SCHEDULE_TICK`] 扫一次到期 Provider（绑定插件 && 启用 && interval > 0 &&
/// 距上次查询已达间隔），串行执行并写入数据库。单个 Provider 失败乃至 panic 都不影响
/// 循环——配额脚本是用户可编辑的，任何一个坏脚本都不该让整个看板停摆。
///
/// `engine` 由调用方提供（与手动刷新共用一个），`plugins_dir` 在此固定到它上面，
/// 避免两个宿主各自维护一份脚本目录来源。
pub fn spawn_quota_scheduler(
    db: Arc<Database>,
    engine: QuotaEngine,
    plugins_dir: Option<PathBuf>,
) -> tokio::task::JoinHandle<()> {
    let engine = engine.with_plugins_dir(plugins_dir);
    tokio::spawn(run_quota_loop(db, engine, SCHEDULE_TICK))
}

/// 调度循环体（`tick` 决定扫描周期，测试可传小值）。
pub async fn run_quota_loop(db: Arc<Database>, engine: QuotaEngine, tick: Duration) {
    loop {
        tokio::time::sleep(tick).await;
        let due = match db.list_quota_due(now_unix()) {
            Ok(providers) => providers,
            Err(e) => {
                tracing::error!(error = %e, "读取到期配额查询失败");
                continue;
            }
        };
        for provider in due {
            // 单个 Provider panic 不得中断调度循环：交给独立 task，panic 收敛为 JoinError
            let engine = engine.clone();
            let key = provider.key.clone();
            let mut tasks = tokio::task::JoinSet::new();
            tasks.spawn(async move { engine.run_provider(&provider).await });
            match tasks.join_next().await {
                Some(Ok(Ok(_))) | None => {}
                Some(Ok(Err(e))) => {
                    tracing::error!(provider = %key, error = %e, "配额结果存储失败");
                }
                Some(Err(e)) => {
                    tracing::error!(provider = %key, error = %e, "配额查询执行失败");
                }
            }
        }
    }
}

#[cfg(test)]
mod db_failure_tests {
    use super::*;
    use moonbridge_store::{PlaintextKey, StoreError};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct DatabaseDirectory(PathBuf);

    impl DatabaseDirectory {
        fn execute(&self, sql: &str) {
            rusqlite::Connection::open(self.0.join("quota.sqlite"))
                .unwrap()
                .execute_batch(sql)
                .unwrap();
        }

        fn reject_result_writes(&self) {
            self.execute(
                "CREATE TRIGGER reject_results BEFORE INSERT ON quota_results
                 BEGIN SELECT RAISE(ABORT, 'synthetic write failure'); END;",
            );
        }
    }

    impl Drop for DatabaseDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn file_database() -> (DatabaseDirectory, Arc<Database>, Provider) {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "moonbridge-quota-db-failure-{}-{timestamp}-{}",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let directory = DatabaseDirectory(directory);
        let db = Database::open_with_key(directory.0.join("quota.sqlite"), Box::new(PlaintextKey))
            .unwrap();
        db.upsert_plugin(&PluginRecord {
            name: "quota-script".into(),
            source: "lua".into(),
            script_ref:
                "MB = {}\nfunction MB.query(ctx) return { status = 'ok', summary = 'queried' } end"
                    .into(),
            enabled: true,
            config: json!({}),
            scopes: vec![],
            capabilities: vec![],
            category: "quota".into(),
            config_schema: serde_json::Value::Null,
        })
        .unwrap();
        let provider = Provider {
            key: "stored-provider".into(),
            endpoints: vec![],
            version: None,
            user_agent: None,
            web_search: None,
            extra: json!({}),
            enabled: true,
            quota_plugin_ref: "quota-script".into(),
            quota_interval_secs: 60,
            quota_enabled: true,
            quota_config: json!({}),
            created_at: 0,
            updated_at: 0,
        };
        db.upsert_provider(&provider).unwrap();
        let stored = db.get_provider(&provider.key).unwrap().unwrap();
        assert_eq!(stored.quota_plugin_ref, provider.quota_plugin_ref);
        assert!(stored.quota_enabled);
        assert!(db.list_quota_results(&provider.key).unwrap().is_empty());
        (directory, Arc::new(db), provider)
    }

    fn assert_sqlite_error(error: StoreError, expected: &str) {
        match error {
            StoreError::Db(error) => assert_eq!(error.to_string(), expected),
            error => panic!("expected SQLite error {expected:?}, got {error:?}"),
        }
    }

    async fn assert_query_succeeds(engine: &QuotaEngine, provider: &Provider) {
        let preview = engine.test_provider(provider).await;
        assert_eq!(preview.len(), 1);
        assert_eq!(preview[0].result.status, "ok");
        assert_eq!(preview[0].result.error, None);
        assert_eq!(
            preview[0].result.payload.as_ref().unwrap()["summary"],
            "queried"
        );
    }

    #[tokio::test]
    async fn refresh_all_propagates_provider_list_failure() {
        let (directory, db, provider) = file_database();
        let engine = QuotaEngine::new(db.clone(), None);
        assert_query_succeeds(&engine, &provider).await;
        directory.execute("DROP TABLE providers;");
        assert_sqlite_error(db.list_providers().unwrap_err(), "no such table: providers");

        assert_sqlite_error(
            engine.refresh_all().await.unwrap_err(),
            "no such table: providers",
        );
        assert!(db.list_quota_results(&provider.key).unwrap().is_empty());
    }

    #[tokio::test]
    async fn refresh_provider_propagates_result_list_failure() {
        let (directory, db, provider) = file_database();
        let engine = QuotaEngine::new(db.clone(), None);
        assert_query_succeeds(&engine, &provider).await;
        directory.execute("DROP TABLE quota_results;");
        assert_sqlite_error(
            db.list_quota_results(&provider.key).unwrap_err(),
            "no such table: quota_results",
        );

        assert_sqlite_error(
            engine.refresh_provider(&provider).await.unwrap_err(),
            "no such table: quota_results",
        );
        assert!(db.get_provider(&provider.key).unwrap().is_some());
    }

    #[tokio::test]
    async fn run_provider_propagates_result_write_failure() {
        let (directory, db, provider) = file_database();
        let engine = QuotaEngine::new(db.clone(), None);
        assert_query_succeeds(&engine, &provider).await;
        directory.reject_result_writes();

        assert_sqlite_error(
            engine.run_provider(&provider).await.unwrap_err(),
            "synthetic write failure",
        );
        assert!(db.list_quota_results(&provider.key).unwrap().is_empty());

        directory.execute("DROP TRIGGER reject_results;");
        let results = engine.run_provider(&provider).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].result.status, "ok");
        assert_eq!(db.list_quota_results(&provider.key).unwrap().len(), 1);
    }

    #[tokio::test]
    async fn refresh_provider_propagates_result_write_failure() {
        let (directory, db, provider) = file_database();
        let engine = QuotaEngine::new(db.clone(), None);
        assert_query_succeeds(&engine, &provider).await;
        directory.reject_result_writes();

        assert_sqlite_error(
            engine.refresh_provider(&provider).await.unwrap_err(),
            "synthetic write failure",
        );
        assert!(db.list_quota_results(&provider.key).unwrap().is_empty());
    }

    #[tokio::test]
    async fn refresh_all_propagates_result_write_failure() {
        let (directory, db, provider) = file_database();
        let engine = QuotaEngine::new(db.clone(), None);
        assert_query_succeeds(&engine, &provider).await;
        directory.reject_result_writes();

        assert_sqlite_error(
            engine.refresh_all().await.unwrap_err(),
            "synthetic write failure",
        );
        assert!(db.list_quota_results(&provider.key).unwrap().is_empty());
        assert_eq!(db.list_providers().unwrap().len(), 1);
    }

    #[test]
    fn builtin_quota_plugins_seed_once_with_manifest_schema() {
        let db = Database::open_in_memory().unwrap();
        assert!(seed_builtin_quota_plugins(&db).unwrap(), "首次应播种");
        let plugins = db.list_plugins().unwrap();
        let seeded: Vec<_> = plugins.iter().filter(|p| p.category == "quota").collect();
        assert_eq!(seeded.len(), BUILTIN_QUOTA_PLUGINS.len());
        for p in &seeded {
            assert!(p.enabled);
            assert!(p.script_ref.contains("MB.query"), "{} 应内联脚本", p.name);
        }
        // 声明了 config_schema 的种子应写入数据库（驱动 Provider 配额配置表单）
        let new_api = db.get_plugin("new-api").unwrap().unwrap();
        assert!(
            new_api.config_schema.get("quota_per_unit").is_some(),
            "new-api 种子应携带 config_schema"
        );
        // 已播种库不再重播；用户删除的内置插件不复活
        db.delete_plugin("new-api").unwrap();
        assert!(!seed_builtin_quota_plugins(&db).unwrap(), "重复调用应跳过");
        assert!(db.get_plugin("new-api").unwrap().is_none());
        // 同名用户插件不被覆盖
        let db2 = Database::open_in_memory().unwrap();
        db2.upsert_plugin(&PluginRecord {
            name: "new-api".to_string(),
            source: "lua".to_string(),
            script_ref: "MB = {}".to_string(),
            enabled: false,
            config: serde_json::Value::Null,
            scopes: Vec::new(),
            capabilities: Vec::new(),
            category: "quota".to_string(),
            config_schema: serde_json::Value::Null,
        })
        .unwrap();
        seed_builtin_quota_plugins(&db2).unwrap();
        assert_eq!(
            db2.get_plugin("new-api").unwrap().unwrap().script_ref,
            "MB = {}"
        );
    }
}
