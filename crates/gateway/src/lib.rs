//! Moon Bridge Next 服务层（网关）。
//!
//! 组合 core/protocol/plugin/store，对外提供内嵌 HTTP 网关：
//! - [`dispatch`]：请求生命周期编排（RAW/CORE 双组钩子串联）。
//! - [`server`]：axum 路由与启动。
//! - [`router`]：模型别名 → provider/上游模型 的路由解析。
//! - [`stream`]：流式 SSE 编排。
//! - [`bridge`]：[`moonbridge_plugin::HostBridge`] 的网关实现。
//! - [`state`]：共享运行状态。
//!
//! 顶层入口：
//! - [`bootstrap`]：由引导配置 + 数据库构造 [`AppState`]（含插件加载）。
//! - [`serve`]：启动网关服务器。
//!
//! 依赖方向：gateway → core, protocol, plugin, store。

pub mod auth;
pub mod quota;
pub mod quota_http;
pub use quota_http::QuotaNetworkPolicy;
pub mod bridge;
pub mod config;
pub mod dispatch;
pub mod error;
pub mod handlers;
pub mod router;
pub mod server;
pub mod session;
pub mod sse;
pub mod state;
pub mod stream;
pub mod trace;
pub mod upstream;
pub mod usage;

use std::sync::Arc;

use moonbridge_plugin::{
    LuaPluginRegistry, LuaRuntime, SandboxLimits, ScopeOverrides, SessionStore,
};
use moonbridge_protocol::{builtin_registry, NoopHooks, PluginHooks, Registry};
use moonbridge_store::Database;

pub use quota::{
    list_views as list_quota_views, run_quota_loop, seed_builtin_quota_plugins,
    spawn_quota_scheduler, view_of as quota_view_of, QuotaEngine, BUILTIN_QUOTA_PLUGINS,
    SCHEDULE_TICK,
};
// 供管理面在保存/导入插件时沙箱求值 `MB` 清单（category/config_schema）。
pub use config::GatewayConfig;
pub use error::{GatewayError, Result};
pub use moonbridge_plugin::LuaRuntime as PluginLuaRuntime;
pub use state::AppState;

use crate::bridge::GatewayBridge;
use crate::upstream::build_client;

/// `max_body_bytes` 直接采用配置（兼现“报文层 body 上限”语义）；`call_timeout`
/// 以上游请求超时为基准再加缓冲，避免误伤插件内合法的 `provider_invoke` 长调用；
/// 指令上限/内存上限/计数步长沿用 [`SandboxLimits::default`]。
fn sandbox_limits(config: &GatewayConfig) -> SandboxLimits {
    SandboxLimits {
        max_body_bytes: config.max_body_bytes,
        call_timeout: std::time::Duration::from_secs(
            config.request_timeout_secs.saturating_add(30),
        ),
        ..SandboxLimits::default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptRef {
    /// 文件脚本：已归一化的路径。
    File(std::path::PathBuf),
    /// 内联 Lua 源码（`script_ref` 原样）。
    Inline(String),
    /// 越出 `plugins_dir` 的文件脚本引用，已拒绝。
    Rejected(String),
}

/// 对可能**尚不存在**的路径做解析：向上找到最近的已存在祖先做 `canonicalize`，
/// 再把剩余片段原样拼回。这样既能校验「UI 新建、还没写入磁盘的子目录脚本」
/// （如 `sub/x.lua`），又能识破中间层的符号链接绕行。
fn canonicalize_pending(path: &std::path::Path) -> Option<std::path::PathBuf> {
    let mut suffix: Vec<std::ffi::OsString> = Vec::new();
    let mut cur = path.to_path_buf();
    loop {
        if cur.exists() {
            let mut c = cur.canonicalize().ok()?;
            for s in suffix.iter().rev() {
                c = c.join(s);
            }
            return Some(c);
        }
        suffix.push(cur.file_name()?.to_owned());
        cur = cur.parent()?.to_path_buf();
    }
}

/// 判断 `path` 是否位于 `root` 之内（含等于 root）。
///
/// 两侧都做 canonicalize（不存在的一侧走 [`canonicalize_pending`]），因此可抵御
/// `../` 上跳与任意层级的符号链接绕行。
pub fn is_within_root(path: &std::path::Path, root: &std::path::Path) -> bool {
    let Some(root_c) = canonicalize_pending(root) else {
        return false;
    };
    match canonicalize_pending(path) {
        Some(c) => c.starts_with(&root_c),
        None => false,
    }
}

/// 解析 `script_ref`：以 `.lua` 结尾（忽略大小写与首尾空白）视为**文件脚本**，
/// 否则视为**内联源码**。
///
/// 这是 gateway 与 app 层**共用的唯一判定**——历史上两侧规则不一致（UI 看 `.lua`
/// 后缀、gateway 看 `Path::exists()`），导致相对路径 `x.lua` 在 UI 里归一到
/// `plugins_dir`、在运行时却按进程 CWD 判断，不一致时被静默当作 Lua 源码编译失败。
///
/// 给了 `plugins_dir` 时，相对路径挂载到其下、绝对路径必须落在其内，越界一律
/// [`ScriptRef::Rejected`]（切断了「把 `script_ref` 写成任意绝对路径 → 让网关去
/// 读/执行该文件」这条逃逸路）。未给 `plugins_dir` 时不做包含性校验（CLI/测试场景）。
pub fn parse_script_ref(script_ref: &str, plugins_dir: Option<&std::path::Path>) -> ScriptRef {
    let trimmed = script_ref.trim();
    if !trimmed.to_ascii_lowercase().ends_with(".lua") {
        return ScriptRef::Inline(script_ref.to_string());
    }
    let raw = std::path::Path::new(trimmed);
    let root = plugins_dir.filter(|d| !d.as_os_str().is_empty());
    let resolved = match (raw.is_absolute(), root) {
        (_, Some(r)) if !raw.is_absolute() => r.join(raw),
        _ => raw.to_path_buf(),
    };
    if let Some(r) = root {
        if !is_within_root(&resolved, r) {
            return ScriptRef::Rejected(trimmed.to_string());
        }
    }
    ScriptRef::File(resolved)
}

/// 从 store 加载 Lua 插件，构造 [`PluginHooks`]。
///
/// 加载条件：插件全局启用，或存在任一维度（route/model/provider/global）
/// enabled=1 的 binding（全局停用但被某作用域强制启用的插件仍需加载）。
/// 运行时过滤：由 [`LuaPluginRegistry`] 按「就近作用域覆盖」逐请求过滤。
/// 无插件或全部加载失败时回退 [`NoopHooks`]。单个插件加载失败仅记录错误，
/// 不影响其它插件与网关启动。
pub fn load_plugins(
    db: &Arc<Database>,
    registry: &Arc<Registry>,
    client: &reqwest::Client,
    limits: SandboxLimits,
    plugins_dir: Option<&std::path::Path>,
) -> Arc<dyn PluginHooks> {
    let sessions = SessionStore::new();
    let bridge = Arc::new(GatewayBridge::new(
        client.clone(),
        db.clone(),
        registry.clone(),
        limits.call_timeout,
    ));
    let mut runtimes: Vec<Arc<LuaRuntime>> = Vec::new();

    // 全维度三态绑定表：plugin_name → 各作用域绑定
    let mut overrides: std::collections::HashMap<String, ScopeOverrides> =
        std::collections::HashMap::new();
    match db.list_bindings_all() {
        Ok(bindings) => {
            for b in bindings {
                overrides.entry(b.plugin_name).or_default().insert(
                    &b.scope,
                    b.scope_key,
                    b.enabled,
                );
            }
        }
        Err(e) => tracing::error!(error = %e, "读取插件绑定失败"),
    }

    match db.list_plugins() {
        Ok(plugins) => {
            for p in plugins {
                // 类别白名单：只有 core 插件进入请求链路注册表；quota 等非核心
                // 类别由各自的引擎驱动，永不在此加载
                if p.category != "core" {
                    continue;
                }
                // 全局停用但被某作用域强制启用的插件仍需加载
                let force_enabled = overrides
                    .get(&p.name)
                    .is_some_and(ScopeOverrides::any_enabled);
                if !p.enabled && !force_enabled {
                    continue;
                }
                let Some(script) = read_script(&p.script_ref, plugins_dir) else {
                    continue; // 具体原因已在 read_script 内记录
                };
                match LuaRuntime::new_with_limits(
                    &p.name,
                    &script,
                    &p.config,
                    bridge.clone(),
                    sessions.clone(),
                    limits.clone(),
                ) {
                    Ok(mut rt) => {
                        rt.enabled = p.enabled;
                        let caps: Vec<String> = rt.manifest.capabilities.iter().cloned().collect();
                        tracing::info!(plugin = %p.name, enabled = p.enabled, capabilities = ?caps, "已加载 Lua 插件");
                        runtimes.push(Arc::new(rt));
                    }
                    Err(e) => tracing::error!(plugin = %p.name, error = %e, "插件加载失败"),
                }
            }
        }
        Err(e) => tracing::error!(error = %e, "读取插件列表失败"),
    }

    if runtimes.is_empty() {
        Arc::new(NoopHooks)
    } else {
        // `sessions` 与所有插件运行时共享：会话淘汰时经它回收 `mb.session` 桶
        Arc::new(LuaPluginRegistry::new(
            runtimes,
            overrides,
            sessions.clone(),
        ))
    }
}

/// 读取插件脚本源码；`None` 表示该插件应被跳过（原因已记录）。
///
/// 判定与包含性校验统一走 [`parse_script_ref`]：文件脚本必须存在且可读；越出
/// `plugins_dir` 的绝对路径直接拒绝；非 `.lua` 的引用按内联源码原样返回。
fn read_script(script_ref: &str, plugins_dir: Option<&std::path::Path>) -> Option<String> {
    match parse_script_ref(script_ref, plugins_dir) {
        ScriptRef::Inline(src) => Some(src),
        ScriptRef::Rejected(p) => {
            tracing::error!(script_ref = %p, "插件脚本路径越出 plugins_dir，拒绝加载");
            None
        }
        ScriptRef::File(path) => {
            if !path.is_file() {
                tracing::error!(path = %path.display(), "插件脚本文件不存在，跳过该插件");
                return None;
            }
            match std::fs::read_to_string(&path) {
                Ok(src) => Some(src),
                Err(e) => {
                    tracing::error!(path = %path.display(), error = %e, "读取插件脚本失败，跳过该插件");
                    None
                }
            }
        }
    }
}

/// 由引导配置 + 数据库构造网关共享状态（装配内置 Adapter、加载插件、构建 HTTP 客户端）。
pub fn bootstrap(config: GatewayConfig, db: Arc<Database>) -> Result<Arc<AppState>> {
    let registry = Arc::new(builtin_registry());
    let client = build_client(&config)?;
    let limits = sandbox_limits(&config);
    let hooks = load_plugins(
        &db,
        &registry,
        &client,
        limits,
        config.plugins_dir.as_deref().map(std::path::Path::new),
    );
    Ok(AppState::new(config, db, registry, hooks, client))
}

/// 启动网关服务器（阻塞直到退出）。
pub async fn serve(config: GatewayConfig, db: Arc<Database>) -> Result<()> {
    let state = bootstrap(config, db)?;
    server::serve(state).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn tempdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mb-scriptref-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("plugins")).unwrap();
        dir
    }

    #[test]
    fn non_lua_ref_is_inline_source() {
        let dir = tempdir("inline");
        let src = "MB = { capabilities = { 'core' } }\nfunction MB.on_request(c, r) return r end";
        assert_eq!(
            parse_script_ref(src, Some(&dir.join("plugins"))),
            ScriptRef::Inline(src.to_string())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn relative_ref_is_rooted_into_plugins_dir() {
        let dir = tempdir("relative");
        let plugins = dir.join("plugins");
        assert_eq!(
            parse_script_ref("demo.lua", Some(&plugins)),
            ScriptRef::File(plugins.join("demo.lua"))
        );
        // 带空白与大小写后缀也应识别为文件脚本
        assert_eq!(
            parse_script_ref("  sub/Demo.LUA  ", Some(&plugins)),
            ScriptRef::File(plugins.join("sub/Demo.LUA"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn absolute_ref_inside_plugins_dir_accepted_outside_rejected() {
        let dir = tempdir("absolute");
        let plugins = dir.join("plugins");
        let inside = plugins.join("ok.lua");
        std::fs::write(&inside, "MB = {}").unwrap();
        assert_eq!(
            parse_script_ref(inside.to_str().unwrap(), Some(&plugins)),
            // 契约是「按传入路径原样返回」——containment 校验在 is_within_root
            // 内部已做 canonicalize；macOS 的 /var 是 /private/var 符号链接，
            // 断言侧 canonicalize 会在该平台凭空制造两个不等的路径。
            ScriptRef::File(inside.clone())
        );

        for evil in [
            "/etc/cron.d/evil.lua",
            "/home/other/.bashrc.lua",
            format!("{}/outside.lua", dir.display()).as_str(),
        ] {
            assert!(
                matches!(
                    parse_script_ref(evil, Some(&plugins)),
                    ScriptRef::Rejected(_)
                ),
                "目录外绝对路径必须被拒绝: {evil}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn dotdot_traversal_rejected() {
        let dir = tempdir("traverse");
        let plugins = dir.join("plugins");
        std::fs::write(dir.join("up.lua"), "MB = {}").unwrap();
        assert!(
            matches!(
                parse_script_ref("../up.lua", Some(&plugins)),
                ScriptRef::Rejected(_)
            ),
            "../ 上跳必须被拒绝"
        );
        assert!(
            matches!(
                parse_script_ref("./a/../../up.lua", Some(&plugins)),
                ScriptRef::Rejected(_)
            ),
            "混合相对片段上跳必须被拒绝"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 符号链接不得成为越狱通道（`resolve_within` 式的 canonicalize 校验是关键）。
    #[cfg(unix)]
    #[test]
    fn symlink_escape_rejected() {
        use std::os::unix::fs::symlink;
        let dir = tempdir("symlink");
        let plugins = dir.join("plugins");
        let outside = dir.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("evil.lua"), "MB = {}").unwrap();
        symlink(outside.join("evil.lua"), plugins.join("link.lua")).unwrap();

        assert!(
            matches!(
                parse_script_ref("link.lua", Some(&plugins)),
                ScriptRef::Rejected(_)
            ),
            "指向目录外的符号链接必须被拒绝"
        );
        // 目录内的真实链接目标仍可用
        std::fs::write(plugins.join("real.lua"), "MB = {}").unwrap();
        assert!(matches!(
            parse_script_ref("real.lua", Some(&plugins)),
            ScriptRef::File(_)
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 中间层是符号链接目录时同样不得越狱（校验必须逐级解析，而非只看末段）。
    #[cfg(unix)]
    #[test]
    fn symlinked_intermediate_dir_rejected() {
        use std::os::unix::fs::symlink;
        let dir = tempdir("symdir");
        let plugins = dir.join("plugins");
        let outside = dir.join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        symlink(&outside, plugins.join("via")).unwrap();

        assert!(
            matches!(
                parse_script_ref("via/evil.lua", Some(&plugins)),
                ScriptRef::Rejected(_)
            ),
            "经符号链接目录上跳必须被拒绝"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 尚未写入磁盘的子目录脚本必须判为合法 `File`——否则 `plugin_write_script`
    /// 永远建不出新子目录（先校验后建目录）。
    #[test]
    fn not_yet_created_nested_path_is_accepted() {
        let dir = tempdir("nested");
        let plugins = dir.join("plugins");
        assert_eq!(
            parse_script_ref("a/b/c.lua", Some(&plugins)),
            ScriptRef::File(plugins.join("a/b/c.lua")),
            "新建多级子目录不应被误判为越界"
        );
        // 但同样不存在的多级上跳仍须拒绝
        assert!(matches!(
            parse_script_ref("a/../../c.lua", Some(&plugins)),
            ScriptRef::Rejected(_)
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 未配置 plugins_dir（CLI/测试）时不做包含性校验，但归一规则保持一致。
    #[test]
    fn without_plugins_dir_no_containment() {
        let dir = tempdir("noroot");
        let f = dir.join("loose.lua");
        std::fs::write(&f, "MB = {}").unwrap();
        assert_eq!(
            parse_script_ref(f.to_str().unwrap(), None),
            ScriptRef::File(f.clone())
        );
        assert_eq!(
            parse_script_ref("rel.lua", None),
            ScriptRef::File(PathBuf::from("rel.lua"))
        );
        // 空字符串目录视作未配置
        assert_eq!(
            parse_script_ref("rel.lua", Some(Path::new(""))),
            ScriptRef::File(PathBuf::from("rel.lua"))
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_file_script_is_skipped_not_inlined() {
        let dir = tempdir("missing");
        let plugins = dir.join("plugins");
        // 旧实现会把 "nope.lua" 当内联源码交给 Lua 编译；现在明确跳过
        assert_eq!(read_script("nope.lua", Some(&plugins)), None);
        // 真实存在的文件脚本正常读出
        std::fs::write(plugins.join("here.lua"), "MB = { x = 1 }").unwrap();
        assert_eq!(
            read_script("here.lua", Some(&plugins)),
            Some("MB = { x = 1 }".to_string())
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
