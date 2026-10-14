//! 配额查询引擎集成测试：真实 Lua 脚本（含 `mb.http`）+ 内存库 + mock 上游。
//!
//! 覆盖写入 `crates/gateway/src/quota.rs` 的四层契约：脚本返回值的归一（quotas
//! camelCase + used/left 互补 + 严格 type 契约）、逐端点拆行（每端点各跑一次、
//! 掩码 label、失败按端点独立分层、prune 残留行）、插件解析（缺失/类别不符），
//! 以及定时调度「从未查询即到期」的选择路径。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::routing::post;
use axum::{Json, Router};
use moonbridge_gateway::{run_quota_loop, QuotaEngine, QuotaNetworkPolicy};
use moonbridge_store::{Database, Endpoint, PluginRecord, Provider, QuotaKeyResult, QuotaResult};
use serde_json::{json, Value};

/// 成功脚本：给出 used_percent，验证引擎补齐 left_percent。
const SCRIPT_OK: &str = r#"
MB = {}
function MB.query(ctx)
  return {
    status = "ok",
    quotas = { { type = "percentage", label = "Weekly", used_percent = 43, reset_at = "t" } },
    summary = "s",
  }
end
"#;

/// 起一个 mock 配额上游（`POST /quota`），返回 `{"left": 57}`；返回其 base_url。
async fn spawn_mock_quota() -> String {
    async fn quota() -> Json<Value> {
        Json(json!({ "left": 57 }))
    }

    let app = Router::new().route("/quota", post(quota));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// 注册一个 quota 类插件（内联脚本）。
fn plugin(db: &Database, name: &str, script: &str) {
    db.upsert_plugin(&PluginRecord {
        name: name.to_string(),
        source: "lua".to_string(),
        script_ref: script.to_string(),
        enabled: true,
        config: Value::Null,
        scopes: vec![],
        capabilities: vec![],
        category: "quota".to_string(),
        config_schema: serde_json::Value::Null,
    })
    .unwrap();
}

/// 构造绑定了配额插件的 Provider：`endpoints` 为 (base_url, api_key) 列表
/// （协议随意，配额查询不读协议）。
fn bound_provider(key: &str, plugin_ref: &str, endpoints: &[(&str, &str)]) -> Provider {
    Provider {
        key: key.to_string(),
        endpoints: endpoints
            .iter()
            .map(|(url, k)| Endpoint {
                protocol: "openai".to_string(),
                base_url: url.to_string(),
                api_key: k.to_string(),
            })
            .collect(),
        version: None,
        user_agent: None,
        web_search: None,
        extra: json!({}),
        enabled: true,
        quota_plugin_ref: plugin_ref.to_string(),
        quota_interval_secs: 60,
        quota_enabled: true,
        quota_config: json!({"widget": "quota"}),
        created_at: 0,
        updated_at: 0,
    }
}

/// 便捷路径：注册 `SCRIPT_OK` 插件并返回绑定的单端点 Provider。
fn ok_provider(db: &Database, key: &str) -> Provider {
    plugin(db, "quota-ok", SCRIPT_OK);
    bound_provider(key, "quota-ok", &[("https://api.example.test", "sk-test")])
}

fn engine(db: &Arc<Database>, plugins_dir: Option<PathBuf>) -> QuotaEngine {
    QuotaEngine::new(db.clone(), plugins_dir)
}

/// 单端点 Provider 的运行结果：引擎返回逐端点数组，单端点场景取 [0]。
fn only(results: &[QuotaKeyResult]) -> &QuotaKeyResult {
    assert_eq!(results.len(), 1, "单端点应只有一行结果: {results:?}");
    &results[0]
}

fn stored(db: &Database, key: &str, idx: i64) -> QuotaKeyResult {
    db.get_quota_key_result(key, idx)
        .unwrap()
        .expect("结果应已写入数据库")
}

fn assert_close(actual: Option<f64>, expected: f64) {
    let v = actual.unwrap_or_else(|| panic!("缺少百分比字段，期望 {expected}"));
    assert!(
        (v - expected).abs() < 1e-9,
        "期望 {expected}，实际 {v}（互补值必须精确补齐）"
    );
}

#[tokio::test]
async fn run_provider_success_normalizes_quotas() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let p = ok_provider(&db, "ok");
    let results = engine(&db, None).run_provider(&p).await.unwrap();
    let res = &only(&results).result;

    assert_eq!(res.status, "ok");
    assert_eq!(res.error, None);
    assert!(res.queried_at > 0);

    let payload = res.payload.clone().expect("成功结果应带 payload");
    assert_eq!(payload["summary"], json!("s"), "自定义字段原样保留");
    let quota = &payload["quotas"][0];
    assert_eq!(quota["label"], json!("Weekly"));
    assert_close(quota["usedPercent"].as_f64(), 43.0);
    assert_close(quota["leftPercent"].as_f64(), 57.0);
    assert_eq!(quota["resetAt"], json!("t"));
    assert!(
        quota.get("used_percent").is_none(),
        "归一后不得留下 snake_case 键: {quota}"
    );
    assert_eq!(
        payload["quotas"].as_array().map(Vec::len),
        Some(1),
        "quotas 不应被改写形状: {payload}"
    );

    // 写入数据库结果与返回值一致（JSON 往返不做数值重排，仅尾数 0 可能消失）
    let roundtrip: Value = serde_json::from_str(&payload.to_string()).unwrap();
    assert_eq!(stored(&db, "ok", 0).result.payload, Some(roundtrip));
}

#[tokio::test]
async fn engine_failure_keeps_previous_payload() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let eng = engine(&db, None);

    let p = ok_provider(&db, "keep");
    let results = eng.run_provider(&p).await.unwrap();
    let first = &only(&results).result;
    assert_eq!(first.status, "ok");
    let good = first.payload.clone().expect("首次成功应有 payload");

    // 换成抛错脚本重跑：脚本读到但立即失败 → 引擎级失败
    plugin(&db, "quota-ok", "error(\"上游不可达\")");
    let results = eng.run_provider(&p).await.unwrap();
    let res = &only(&results).result;
    assert_eq!(res.status, "error");
    assert!(res.error.as_deref().is_some_and(|e| !e.is_empty()));
    assert_eq!(
        res.payload,
        Some(good.clone()),
        "引擎级失败必须保留上一次的 payload"
    );
    assert_eq!(stored(&db, "keep", 0).result.payload, Some(good));
    assert_eq!(stored(&db, "keep", 0).result.status, "error");
}

#[tokio::test]
async fn business_error_updates_payload() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    plugin(
        &db,
        "quota-biz",
        r#"
        MB = {}
        function MB.query(ctx)
          return { status = "error", message = "key 失效", note = "diag" }
        end
        "#,
    );
    let p = bound_provider("biz", "quota-biz", &[("https://a", "sk-t")]);
    let results = engine(&db, None).run_provider(&p).await.unwrap();
    let res = &only(&results).result;
    assert_eq!(res.status, "error");
    assert_eq!(
        res.error.as_deref(),
        Some("key 失效"),
        "error 取脚本 message"
    );
    let payload = res
        .payload
        .clone()
        .expect("脚本返回了 table，payload 应更新");
    assert_eq!(payload["note"], json!("diag"), "诊断字段必须保留");
}

#[tokio::test]
async fn missing_query_entry_is_engine_error() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    plugin(
        &db,
        "quota-noquery",
        "MB = {}\nfunction MB.other(ctx) return {} end",
    );
    let p = bound_provider("noquery", "quota-noquery", &[("https://a", "sk-t")]);
    let results = engine(&db, None).run_provider(&p).await.unwrap();
    let res = &only(&results).result;
    assert_eq!(res.status, "error");
    assert_eq!(res.error.as_deref(), Some("配额脚本执行失败"));
    assert_eq!(res.payload, None, "首次失败无可保留的旧 payload");
}

#[tokio::test]
async fn unbound_and_wrong_category_plugins_are_engine_errors() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let eng = engine(&db, None);

    // 未绑定插件
    let mut p = bound_provider("unbound", "", &[("https://a", "sk-t")]);
    let results = eng.run_provider(&p).await.unwrap();
    let res = &only(&results).result;
    assert_eq!(res.status, "error");
    assert_eq!(res.error.as_deref(), Some("未绑定配额查询插件"));

    // 引用了不存在的插件
    p.quota_plugin_ref = "ghost".to_string();
    let results = eng.run_provider(&p).await.unwrap();
    assert_eq!(
        only(&results).result.error.as_deref(),
        Some("配额插件 ghost 不存在")
    );

    // 引用了 core 类插件（类别不符，拒绝进入配额引擎）
    db.upsert_plugin(&PluginRecord {
        name: "core-plugin".to_string(),
        source: "lua".to_string(),
        script_ref: SCRIPT_OK.to_string(),
        enabled: true,
        config: Value::Null,
        scopes: vec![],
        capabilities: vec![],
        category: "core".to_string(),
        config_schema: serde_json::Value::Null,
    })
    .unwrap();
    p.quota_plugin_ref = "core-plugin".to_string();
    let results = eng.run_provider(&p).await.unwrap();
    assert_eq!(
        only(&results).result.error.as_deref(),
        Some("插件 core-plugin 不是配额查询类插件")
    );

    // 解析失败路径同样写单行结果并移除残留
    db.upsert_quota_key_result(
        "unbound",
        &QuotaKeyResult {
            key_index: 3,
            key_label: "old".to_string(),
            result: QuotaResult {
                status: "ok".to_string(),
                payload: Some(json!({"summary": "stale"})),
                error: None,
                queried_at: 1,
            },
        },
    )
    .unwrap();
    let results = eng.run_provider(&p).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(
        db.list_quota_results("unbound").unwrap().len(),
        1,
        "解析失败路径同样 prune 残留行"
    );
}

#[tokio::test]
async fn mb_http_real_roundtrip_complements_percent() {
    let base_url = spawn_mock_quota().await;
    let db = Arc::new(Database::open_in_memory().unwrap());
    let script = r#"
    MB = {}
    function MB.query(ctx)
      local r = mb.http.request({
        method = "POST",
        url = ctx.base_url .. "/quota",
        headers = { { "Authorization", "Bearer " .. ctx.key } },
        timeout_ms = 5000,
      })
      if r.status ~= 200 then
        return { status = "error", message = "上游 " .. tostring(r.status) }
      end
      return { status = "ok", quotas = { { type = "percentage", label = "5h", left_percent = r.body.left } } }
    end
    "#;
    plugin(&db, "quota-http", script);
    let p = bound_provider("http", "quota-http", &[(&base_url, "sk-test")]);

    let policy = QuotaNetworkPolicy::from_config(vec![base_url], None).unwrap();
    let results = engine(&db, None)
        .with_network_policy(policy)
        .run_provider(&p)
        .await
        .unwrap();
    let res = &only(&results).result;
    assert_eq!(res.status, "ok", "error: {:?}", res.error);
    let quota = &res.payload.clone().expect("应有 payload")["quotas"][0];
    assert_eq!(quota["label"], json!("5h"));
    assert_close(quota["leftPercent"].as_f64(), 57.0);
    assert_close(quota["usedPercent"].as_f64(), 43.0);
}

#[tokio::test]
async fn script_ref_outside_plugins_dir_is_rejected() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let dir = std::env::temp_dir().join(format!("moonbridge-quota-plugins-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // 越界文件真实存在：拒绝必须来自路径包含性校验，而不是「读不到文件」
    let outside = std::env::temp_dir().join(format!(
        "moonbridge-quota-outside-{}.lua",
        std::process::id()
    ));
    std::fs::write(&outside, SCRIPT_OK).unwrap();

    plugin(&db, "quota-escape", outside.to_str().unwrap());
    let p = bound_provider("escape", "quota-escape", &[("https://a", "sk-t")]);
    let results = engine(&db, Some(dir.clone()))
        .run_provider(&p)
        .await
        .unwrap();
    let res = &only(&results).result;
    assert_eq!(res.status, "error");
    assert!(
        res.error
            .as_deref()
            .is_some_and(|e| e.contains("越出插件目录")),
        "应报路径越界: {:?}",
        res.error
    );

    let _ = std::fs::remove_file(&outside);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn scheduler_picks_up_never_queried_provider() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let p = ok_provider(&db, "sched");
    db.upsert_provider(&p).unwrap();
    assert!(
        db.list_quota_results("sched").unwrap().is_empty(),
        "调度前不应有结果行"
    );

    let task = tokio::spawn(run_quota_loop(
        db.clone(),
        engine(&db, None),
        Duration::from_millis(50),
    ));

    let mut landed = None;
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        let rows = db.list_quota_results("sched").unwrap();
        if let Some(r) = rows.into_iter().find(|r| r.result.status == "ok") {
            landed = Some(r);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    task.abort();

    let r = landed.expect("从未查询的启用 Provider 应在超时前被调度并写入数据库");
    assert_eq!(
        r.result.payload.expect("应有 payload")["summary"],
        json!("s")
    );
}

#[tokio::test]
async fn quota_type_entries_pass_through_without_percent() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let script = r#"
    MB = {}
    function MB.query(ctx)
      return {
        status = "ok",
        quotas = {
          { type = "quota", label = "余额", unit = "¥", used_amount = 12.5, left_amount = 37.5 },
          { type = "quota", label = "流量", unit = "GB", used_amount = 3 },
        },
      }
    end
    "#;
    plugin(&db, "quota-amount", script);
    let p = bound_provider("amount", "quota-amount", &[("https://a", "sk-t")]);
    let results = engine(&db, None).run_provider(&p).await.unwrap();
    let res = &only(&results).result;
    assert_eq!(res.status, "ok", "error: {:?}", res.error);

    let payload = res.payload.clone().expect("成功结果应带 payload");
    let q0 = &payload["quotas"][0];
    assert_eq!(q0["label"], json!("余额"));
    assert_eq!(q0["unit"], json!("¥"));
    assert_close(q0["usedAmount"].as_f64(), 12.5);
    assert_close(q0["leftAmount"].as_f64(), 37.5);
    assert!(
        q0.get("usedPercent").is_none() && q0.get("leftPercent").is_none(),
        "金额模式不得生成 percent 字段: {q0}"
    );
    assert!(
        q0.get("used_amount").is_none(),
        "归一后不得留下 snake_case 键: {q0}"
    );

    // 只有 used_amount 时不得反向推出 left_amount 或 percent
    let q1 = &payload["quotas"][1];
    assert_close(q1["usedAmount"].as_f64(), 3.0);
    assert!(
        q1.get("leftAmount").is_none()
            && q1.get("usedPercent").is_none()
            && q1.get("leftPercent").is_none(),
        "amount 之间及与 percent 之间均不做互补互推: {q1}"
    );

    let roundtrip: Value = serde_json::from_str(&payload.to_string()).unwrap();
    assert_eq!(stored(&db, "amount", 0).result.payload, Some(roundtrip));
}

/// 严格契约：缺 type / type 非法 / 类型必填字段缺失的配额被剔除并记 warnings，
/// 合法元素不受影响；quotas 数组本身形状不变（仍是数组）。
#[tokio::test]
async fn invalid_quota_entries_are_dropped_with_warnings() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let script = r#"
    MB = {}
    function MB.query(ctx)
      return {
        status = "ok",
        quotas = {
          { type = "percentage", label = "正常", used_percent = 43 },
          { label = "缺 type", used_percent = 10 },
          { type = "wat", label = "非法 type", used_percent = 10 },
          { type = "quota", label = "缺金额" },
          { type = "counter", label = "计数", used_amount = 42 },
        },
      }
    end
    "#;
    plugin(&db, "quota-bad", script);
    let p = bound_provider("bad", "quota-bad", &[("https://a", "sk-t")]);
    let results = engine(&db, None).run_provider(&p).await.unwrap();
    let res = &only(&results).result;
    assert_eq!(res.status, "ok", "error: {:?}", res.error);
    let payload = res.payload.clone().expect("成功结果应带 payload");
    let quotas = payload["quotas"].as_array().unwrap();
    assert_eq!(quotas.len(), 2, "只保留合法配额: {payload}");
    assert_eq!(quotas[0]["label"], json!("正常"));
    assert_eq!(quotas[1]["label"], json!("计数"));
    let warnings = payload["warnings"].as_array().expect("应记录剔除原因");
    assert_eq!(warnings.len(), 3, "{payload}");
}

#[tokio::test]
async fn provider_endpoints_split_into_per_key_results() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    // 脚本按单 key 编写：summary 回显 ctx.key 与 ctx 形状，便于逐端点断言
    let script = r#"
    MB = {}
    function MB.query(ctx)
      return {
        status = "ok",
        quotas = {},
        summary = ctx.key .. "|" .. ctx.keys[1]
          .. "|" .. ctx.base_url
          .. "|" .. tostring(#ctx.keys) .. "|" .. ctx.provider
          .. "|" .. tostring(ctx.base_urls == nil and ctx.endpoints == nil),
      }
    end
    "#;
    plugin(&db, "quota-echo", script);
    // 端点：sk-alpha-0001 → 留空（沿用前一个）→ sk-beta-00002 → sk-alpha-0001（重复）
    let p = bound_provider(
        "kimi",
        "quota-echo",
        &[
            ("https://a", "sk-alpha-0001"),
            ("https://b", ""),
            ("https://a2", "sk-beta-00002"),
            ("https://c", "sk-alpha-0001"),
        ],
    );

    let results = engine(&db, None).run_provider(&p).await.unwrap();
    assert_eq!(results.len(), 4, "每个端点各拆一行（不去重）: {results:?}");

    let first = &results[0];
    assert_eq!(first.key_index, 0);
    assert_eq!(first.key_label, "sk-alp…0001", "key_label 是掩码而非原文");
    assert_eq!(first.result.status, "ok", "error: {:?}", first.result.error);
    assert_eq!(
        first.result.payload.as_ref().unwrap()["summary"],
        json!("sk-alpha-0001|sk-alpha-0001|https://a|1|kimi|true"),
        "ctx 为单 key 形状、base_url 取该端点、provider 为 key"
    );

    // 空 api_key 端点沿用上一个非空 key，base_url 仍是自己的
    let second = &results[1];
    assert_eq!(second.key_index, 1);
    assert_eq!(
        second.result.payload.as_ref().unwrap()["summary"],
        json!("sk-alpha-0001|sk-alpha-0001|https://b|1|kimi|true")
    );

    let third = &results[2];
    assert_eq!(third.key_label, "sk-bet…0002");
    assert_eq!(
        third.result.payload.as_ref().unwrap()["summary"],
        json!("sk-beta-00002|sk-beta-00002|https://a2|1|kimi|true")
    );

    // 重复 key 不去重：端点是独立的查询目标
    assert_eq!(
        results[3].result.payload.as_ref().unwrap()["summary"],
        json!("sk-alpha-0001|sk-alpha-0001|https://c|1|kimi|true")
    );

    // 写入数据库行与返回值一致；结果表里没有 key 原文
    let rows = db.list_quota_results("kimi").unwrap();
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0].key_label, "sk-alp…0001");
    assert_eq!(rows[2].key_label, "sk-bet…0002");
}

#[tokio::test]
async fn per_key_failure_is_independent() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    // 只对第二个 key 抛错：引擎级失败按端点独立，不影响其它行
    let script = r#"
    MB = {}
    function MB.query(ctx)
      if ctx.key == "sk-beta-00002" then
        error("boom")
      end
      return { status = "ok", quotas = {}, summary = ctx.key }
    end
    "#;
    plugin(&db, "quota-flaky", script);
    let p = bound_provider(
        "kimi",
        "quota-flaky",
        &[
            ("https://a", "sk-alpha-0001"),
            ("https://b", "sk-beta-00002"),
        ],
    );

    let results = engine(&db, None).run_provider(&p).await.unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].result.status, "ok");
    assert_eq!(
        results[0].result.payload.as_ref().unwrap()["summary"],
        json!("sk-alpha-0001")
    );
    assert_eq!(results[1].result.status, "error");
    assert_eq!(
        results[1].result.error.as_deref(),
        Some("配额脚本执行失败"),
        "执行错误必须脱敏，不返回脚本原始消息"
    );
    assert_eq!(
        results[1].result.payload, None,
        "首次失败无可保留的旧 payload"
    );
}

#[tokio::test]
async fn engine_failure_keeps_payload_per_key() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    plugin(&db, "quota-swap", SCRIPT_OK);
    let p = bound_provider(
        "kimi",
        "quota-swap",
        &[
            ("https://a", "sk-alpha-0001"),
            ("https://b", "sk-beta-00002"),
        ],
    );
    let eng = engine(&db, None);

    let results = eng.run_provider(&p).await.unwrap();
    assert_eq!(results.len(), 2);
    let good1 = results[1].result.payload.clone().expect("key1 首次成功");

    // 换成「只对 key1 抛错」的脚本：key0 照常更新，key1 保留旧 payload
    plugin(
        &db,
        "quota-swap",
        r#"
    MB = {}
    function MB.query(ctx)
      if ctx.key == "sk-beta-00002" then error("boom") end
      return { status = "ok", quotas = {}, summary = "new-" .. ctx.key }
    end
    "#,
    );
    let results = eng.run_provider(&p).await.unwrap();
    assert_eq!(results[0].result.status, "ok");
    assert_eq!(
        results[0].result.payload.as_ref().unwrap()["summary"],
        json!("new-sk-alpha-0001"),
        "成功的 key 正常更新 payload"
    );
    assert_eq!(results[1].result.status, "error");
    assert_eq!(
        results[1].result.payload,
        Some(good1.clone()),
        "失败的 key 保留自己上一次的 payload"
    );
    assert_eq!(stored(&db, "kimi", 0).result.status, "ok");
    assert_eq!(stored(&db, "kimi", 1).result.payload, Some(good1));
}

#[tokio::test]
async fn shrinking_endpoints_prunes_stale_rows() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    plugin(&db, "quota-ok", SCRIPT_OK);
    let mut p = bound_provider(
        "kimi",
        "quota-ok",
        &[
            ("https://a", "sk-alpha-0001"),
            ("https://b", "sk-beta-00002"),
        ],
    );
    db.upsert_provider(&p).unwrap();
    let eng = engine(&db, None);
    assert_eq!(eng.run_provider(&p).await.unwrap().len(), 2);

    // Provider 删到只剩一个端点：重跑后 idx1 残留行被删除
    p.endpoints.truncate(1);
    db.upsert_provider(&p).unwrap();
    let results = eng.run_provider(&p).await.unwrap();
    assert_eq!(results.len(), 1);
    let rows = db.list_quota_results("kimi").unwrap();
    assert_eq!(rows.len(), 1, "残留行应被 prune: {rows:?}");
    assert_eq!(rows[0].key_index, 0);
}

#[tokio::test]
async fn refresh_provider_reruns_all_endpoints() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    plugin(&db, "quota-ok", SCRIPT_OK);
    let p = bound_provider(
        "one",
        "quota-ok",
        &[
            ("https://a", "sk-alpha-0001"),
            ("https://b", "sk-beta-00002"),
        ],
    );
    db.upsert_provider(&p).unwrap();
    let eng = engine(&db, None);
    eng.run_provider(&p).await.unwrap();

    // 换脚本：key0 报错、key1 返回新 summary。刷新重跑全部端点
    plugin(
        &db,
        "quota-ok",
        r#"
    MB = {}
    function MB.query(ctx)
      if ctx.key == "sk-alpha-0001" then error("down") end
      return { status = "ok", quotas = {}, summary = "changed" }
    end
    "#,
    );
    let view = eng.refresh_provider(&p).await.unwrap();
    assert_eq!(view.provider_key, "one");
    assert_eq!(view.results.len(), 2);
    assert_eq!(view.results[0].result.status, "error");
    assert_eq!(
        view.results[0].result.error.as_deref(),
        Some("配额脚本执行失败"),
        "idx0 必须重跑为脱敏错误"
    );
    assert_eq!(view.results[1].result.status, "ok");
    assert_eq!(
        view.results[1].result.payload.as_ref().unwrap()["summary"],
        json!("changed"),
        "idx1 也随刷新重跑"
    );
}

#[tokio::test]
async fn provider_without_endpoints_runs_once() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    // 无端点 Provider 退化为单次空 key 查询（账号级接口场景）
    plugin(
        &db,
        "quota-nokey",
        r#"
    MB = {}
    function MB.query(ctx)
      return { status = "ok", quotas = {}, summary = "key=" .. ctx.key .. "/n=" .. tostring(#ctx.keys) }
    end
    "#,
    );
    let p = bound_provider("nokey", "quota-nokey", &[]);
    let results = engine(&db, None).run_provider(&p).await.unwrap();
    let kr = only(&results);
    assert_eq!(kr.key_label, "", "空 key 的 label 为空串");
    assert_eq!(kr.result.status, "ok", "error: {:?}", kr.result.error);
    assert_eq!(
        kr.result.payload.as_ref().unwrap()["summary"],
        json!("key=/n=1")
    );
}

#[tokio::test]
async fn single_endpoint_ctx_shape() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let script = r#"
    MB = {}
    function MB.query(ctx)
      return {
        status = "ok",
        quotas = {},
        summary = tostring(#ctx.keys) .. "/" .. ctx.keys[1]
          .. "/" .. ctx.base_url .. "|" .. ctx.provider,
      }
    end
    "#;
    plugin(&db, "quota-shape", script);
    let p = bound_provider(
        "legacy",
        "quota-shape",
        &[("https://api.example.test", "sk-test")],
    );
    let results = engine(&db, None).run_provider(&p).await.unwrap();
    let kr = only(&results);
    assert_eq!(kr.key_label, "sk…", "短 key 掩码为前 2 位 + …");
    assert_eq!(kr.result.status, "ok", "error: {:?}", kr.result.error);
    let payload = kr.result.payload.clone().expect("应有 payload");
    let summary = payload["summary"].as_str().unwrap();
    assert_eq!(
        summary, "1/sk-test/https://api.example.test|legacy",
        "ctx 恒为单元素 keys，base_url 取端点地址: {summary}"
    );
}

#[tokio::test]
async fn reset_at_accepts_number_and_string() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let script = r#"
    MB = {}
    function MB.query(ctx)
      return {
        status = "ok",
        quotas = {
          { type = "percentage", label = "周窗口", used_percent = 43, reset_at = 1789616942 },
          { type = "percentage", label = "月额度", used_percent = 10, reset_at = "每月 1 日" },
          { type = "percentage", label = "坏值", used_percent = 5, reset_at = true },
        },
      }
    end
    "#;
    plugin(&db, "quota-reset", script);
    let p = bound_provider("reset", "quota-reset", &[("https://a", "sk-t")]);
    let results = engine(&db, None).run_provider(&p).await.unwrap();
    let res = &only(&results).result;
    assert_eq!(res.status, "ok", "error: {:?}", res.error);
    let quotas = res.payload.as_ref().unwrap()["quotas"]
        .as_array()
        .unwrap()
        .clone();
    // 数字（unix 秒）归一为字符串，前端据此按时间戳格式化
    assert_eq!(quotas[0]["resetAt"], json!("1789616942"));
    // 字符串原样透传
    assert_eq!(quotas[1]["resetAt"], json!("每月 1 日"));
    // 非法类型（bool）按 None 处理，不使整条配额失效
    assert!(quotas[2].get("resetAt").is_none(), "{:?}", quotas[2]);
    assert_close(quotas[2]["leftPercent"].as_f64(), 95.0);
}

#[tokio::test]
async fn test_provider_runs_without_persisting() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let eng = engine(&db, None);

    // 成功 dry-run：返回本次结果但不写入数据库
    let p = ok_provider(&db, "dry");
    let results = eng.test_provider(&p).await;
    let res = &only(&results).result;
    assert_eq!(res.status, "ok", "error: {:?}", res.error);
    assert!(res.payload.is_some());
    assert!(res.queried_at > 0);
    assert!(
        db.list_quota_results("dry").unwrap().is_empty(),
        "dry-run 不得把结果写数据库"
    );

    // 引擎级失败的 dry-run：payload 为 None（不读历史、不保留旧值）
    plugin(&db, "quota-ok", "error(\"上游不可达\")");
    let results = eng.test_provider(&p).await;
    let res = &only(&results).result;
    assert_eq!(res.status, "error");
    assert!(res.error.as_deref().is_some_and(|e| !e.is_empty()));
    assert_eq!(res.payload, None, "dry-run 引擎级失败 payload 必须为 None");
    assert!(db.list_quota_results("dry").unwrap().is_empty());

    // 多端点 Provider 的 dry-run：逐端点返回，同样不写入数据库
    plugin(&db, "quota-ok", SCRIPT_OK);
    let p = bound_provider(
        "drymulti",
        "quota-ok",
        &[
            ("https://a", "sk-alpha-0001"),
            ("https://b", "sk-beta-00002"),
        ],
    );
    let results = eng.test_provider(&p).await;
    assert_eq!(results.len(), 2, "dry-run 也按端点拆结果");
    assert_eq!(results[0].key_label, "sk-alp…0001");
    assert_eq!(results[1].key_label, "sk-bet…0002");
    assert!(db.list_quota_results("drymulti").unwrap().is_empty());
}

async fn run_builtin_template(
    name: &str,
    status: u16,
    body: Value,
    extra: Value,
) -> QuotaKeyResult {
    let script = match name {
        "NEW_API" => include_str!("../../../plugins/quota/new-api.lua"),
        "KIMI_CODING" => include_str!("../../../plugins/quota/kimi-coding.lua"),
        "COMMANDCODE" => include_str!("../../../plugins/quota/commandcode.lua"),
        "SUB2API" => include_str!("../../../plugins/quota/sub2api.lua"),
        _ => panic!("unknown template"),
    };
    let path = match name {
        "NEW_API" => "/api/user/self",
        "KIMI_CODING" => "/coding/v1/usages",
        "COMMANDCODE" => "/alpha/billing/credits",
        "SUB2API" => "/v1/usage",
        _ => panic!("unknown template"),
    };
    let require_cli = matches!(name, "KIMI_CODING" | "COMMANDCODE");
    let require_json = name == "COMMANDCODE";
    let app = Router::new().route(
        path,
        axum::routing::get(move |headers: axum::http::HeaderMap| {
            let body = body.clone();
            async move {
                assert_eq!(headers.get("authorization").unwrap(), "Bearer sk-test");
                if require_cli {
                    assert_eq!(headers.get("user-agent").unwrap(), "cli");
                }
                if require_json {
                    assert_eq!(headers.get("content-type").unwrap(), "application/json");
                }
                (
                    axum::http::StatusCode::from_u16(status).unwrap(),
                    Json(body),
                )
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let db = Arc::new(Database::open_in_memory().unwrap());
    plugin(&db, "quota-tpl", script);
    let base_url = if name == "COMMANDCODE" {
        format!("http://{addr}{path}")
    } else if name == "SUB2API" {
        format!("  http://{addr}/v1/  ")
    } else {
        format!("http://{addr}/")
    };
    let mut p = bound_provider("template", "quota-tpl", &[(&base_url, "sk-test")]);
    p.quota_config = extra;
    let policy = QuotaNetworkPolicy::from_config(vec![format!("http://{addr}")], None).unwrap();
    let results = engine(&db, None)
        .with_network_policy(policy)
        .test_provider(&p)
        .await;
    server.abort();
    only(&results).clone()
}

#[tokio::test]
async fn new_api_template_uses_remaining_quota_and_default_dollar_conversion() {
    for quota in [json!(12500000), json!("12500000")] {
        let result = run_builtin_template(
            "NEW_API",
            200,
            json!({"data": {"quota": quota, "used_quota": 900000000}}),
            json!({}),
        )
        .await;
        assert_eq!(result.result.status, "ok");
        let payload = result.result.payload.unwrap();
        assert_eq!(payload["quotas"].as_array().unwrap().len(), 1);
        assert_eq!(payload["quotas"][0]["unit"], json!("$"));
        assert_close(payload["quotas"][0]["leftAmount"].as_f64(), 25.0);
    }
    let result = run_builtin_template(
        "NEW_API",
        200,
        json!({"data": {"quota": "12500000"}}),
        json!({"quota_per_unit": 10000000, "unit": "¥"}),
    )
    .await;
    assert_eq!(result.result.status, "ok");
    let payload = result.result.payload.unwrap();
    assert_close(payload["quotas"][0]["leftAmount"].as_f64(), 1.25);
    assert_eq!(payload["quotas"][0]["unit"], json!("¥"));
}

#[tokio::test]
async fn new_api_template_nonpositive_quota_is_an_explicit_empty_success() {
    for quota in [json!(0), json!("0"), json!(-1), json!("-500000")] {
        let result =
            run_builtin_template("NEW_API", 200, json!({"data": {"quota": quota}}), json!({}))
                .await;
        assert_eq!(result.result.status, "ok");
        assert_eq!(result.result.error, None);
        let payload = result.result.payload.unwrap();
        assert_eq!(payload["summary"], json!("无额度记录"));
        assert_eq!(payload["quotas"], json!([]));
    }
}

#[tokio::test]
async fn new_api_template_rejects_missing_remaining_quota_and_invalid_conversion() {
    for data in [
        json!({"used_quota": 12500000}),
        json!({"quota": true}),
        json!({"quota": "nan"}),
        json!({"quota": "inf"}),
    ] {
        let result = run_builtin_template("NEW_API", 200, json!({"data": data}), json!({})).await;
        assert_eq!(result.result.status, "error");
        assert_eq!(
            result.result.error.as_deref(),
            Some("响应缺少有效的 data.quota")
        );
    }
    for conversion in [json!(0), json!(-1), json!(true), json!("nan"), json!("inf")] {
        let result = run_builtin_template(
            "NEW_API",
            200,
            json!({"data": {"quota": 500000}}),
            json!({"quota_per_unit": conversion}),
        )
        .await;
        assert_eq!(result.result.status, "error");
        assert_eq!(
            result.result.error.as_deref(),
            Some("quota_per_unit 必须是大于 0 的有限数字")
        );
    }
}

#[tokio::test]
async fn kimi_template_uses_ratios_and_reset_times() {
    let result = run_builtin_template(
        "KIMI_CODING",
        200,
        json!({"usages": {
            "limit_5h": {"used_ratio": 0, "reset_time": "2026-09-17T12:00:00Z"},
            "limit_7d": {"used_ratio": 0.375, "reset_time": "2026-09-21T12:00:00Z"}
        }}),
        json!({}),
    )
    .await;
    assert_eq!(result.result.status, "ok");
    let payload = result.result.payload.unwrap();
    let quotas = payload["quotas"].as_array().unwrap();
    assert_eq!(quotas.len(), 2);
    assert_close(quotas[0]["usedPercent"].as_f64(), 0.0);
    assert_close(quotas[0]["leftPercent"].as_f64(), 100.0);
    assert_close(quotas[1]["usedPercent"].as_f64(), 37.5);
    assert_eq!(quotas[0]["resetAt"], json!("2026-09-17T12:00:00Z"));
    assert_eq!(quotas[1]["resetAt"], json!("2026-09-21T12:00:00Z"));
}

#[tokio::test]
async fn commandcode_template_uses_monthly_and_window_amounts() {
    let result = run_builtin_template(
        "COMMANDCODE",
        200,
        json!({
            "credits": {"monthlyCredits": 52.345},
            "windowLimits": {
                "fiveHour": {"used": 0, "cap": 14, "resetAt": 1800000000123_i64},
                "weekly": {"used": 36, "cap": 35}
            }
        }),
        json!({}),
    )
    .await;
    assert_eq!(result.result.status, "ok");
    let payload = result.result.payload.unwrap();
    let quotas = payload["quotas"].as_array().unwrap();
    assert_eq!(quotas.len(), 3);
    assert_close(quotas[0]["leftAmount"].as_f64(), 52.35);
    assert_close(quotas[0]["usedAmount"].as_f64(), 17.66);
    assert_eq!(quotas[0]["unit"], json!(""));
    assert_close(quotas[1]["leftAmount"].as_f64(), 14.0);
    assert_eq!(quotas[1]["resetAt"], json!("1800000000"));
    assert_close(quotas[2]["leftAmount"].as_f64(), 0.0);
    let result = run_builtin_template(
        "COMMANDCODE",
        200,
        json!({"credits": {"monthlyCredits": 0}}),
        json!({"monthly_cap": 100, "unit": "$"}),
    )
    .await;
    let payload = result.result.payload.unwrap();
    assert_close(payload["quotas"][0]["usedAmount"].as_f64(), 100.0);
    assert_close(payload["quotas"][0]["leftAmount"].as_f64(), 0.0);
    assert_eq!(payload["quotas"][0]["unit"], json!("$"));
    let result = run_builtin_template(
        "COMMANDCODE",
        200,
        json!({"windowLimits": {"weekly": {"used": 1, "cap": 2}}}),
        json!({}),
    )
    .await;
    assert_eq!(result.result.status, "ok");
    assert!(result.result.payload.unwrap()["summary"].is_null());
}

#[tokio::test]
async fn builtin_templates_report_http_and_schema_errors() {
    for name in ["NEW_API", "KIMI_CODING", "COMMANDCODE", "SUB2API"] {
        for (status, body) in [
            (401, json!({})),
            (200, json!({})),
            (200, json!("not an object")),
        ] {
            let result = run_builtin_template(name, status, body, json!({})).await;
            assert_eq!(result.result.status, "error", "{name}");
            let payload = result
                .result
                .payload
                .expect("business error must retain its payload");
            let message = payload["message"].as_str().unwrap();
            assert!(!message.is_empty());
            assert_eq!(result.result.error.as_deref(), Some(message));
        }
    }
    for (name, body, extra) in [
        (
            "NEW_API",
            json!({"success": false, "message": "access denied"}),
            json!({}),
        ),
        ("NEW_API", json!({}), json!({"quota_per_unit": 0})),
        (
            "KIMI_CODING",
            json!({"usages": {"limit_5h": {"used_ratio": -1}, "limit_7d": {"used_ratio": "invalid"}}}),
            json!({}),
        ),
        (
            "COMMANDCODE",
            json!({"windowLimits": {"weekly": {"used": 1}}}),
            json!({}),
        ),
        ("COMMANDCODE", json!({}), json!({"monthly_cap": 0})),
    ] {
        let result = run_builtin_template(name, 200, body, extra).await;
        assert_eq!(result.result.status, "error", "{name}");
        let payload = result
            .result
            .payload
            .expect("business error must retain its payload");
        assert_eq!(result.result.error.as_deref(), payload["message"].as_str());
        assert!(!payload["message"].as_str().unwrap().is_empty());
    }
}

#[tokio::test]
async fn sub2api_template_preserves_balance_precedence_and_windows() {
    for (body, label, amount) in [
        (
            json!({"quota": {"remaining": "0"}, "balance": 99, "remaining": 100}),
            "Key 剩余额度",
            0.0,
        ),
        (
            json!({"quota": {}, "balance": "12.5", "remaining": 100}),
            "钱包余额",
            12.5,
        ),
        (
            json!({"mode": "quota_limited", "remaining": 3}),
            "Key 剩余额度",
            3.0,
        ),
        (
            json!({"planName": "钱包余额", "remaining": -1}),
            "钱包余额",
            -1.0,
        ),
        (
            json!({"planName": "Pro", "remaining": 7}),
            "Pro 剩余额度",
            7.0,
        ),
        (json!({"remaining": 8}), "剩余额度", 8.0),
    ] {
        let result = run_builtin_template("SUB2API", 200, body, json!({})).await;
        assert_eq!(result.result.status, "ok");
        let payload = result.result.payload.unwrap();
        let quotas = payload["quotas"].as_array().unwrap();
        assert_eq!(quotas.len(), 1);
        assert_eq!(quotas[0]["label"], json!(label));
        assert_eq!(quotas[0]["unit"], json!("$"));
        assert_close(quotas[0]["leftAmount"].as_f64(), amount);
    }
    let result = run_builtin_template(
        "SUB2API",
        200,
        json!({"rate_limits": [
            {"window": "5h", "remaining": 0},
            {"window": "1d", "remaining": "1.5"},
            {"window": "7d", "remaining": 2},
            {"window": "30d", "remaining": 3},
            {"window": "bad", "remaining": "invalid"},
            "invalid"
        ]}),
        json!({}),
    )
    .await;
    assert_eq!(result.result.status, "ok");
    let payload = result.result.payload.unwrap();
    let quotas = payload["quotas"].as_array().unwrap();
    assert_eq!(quotas.len(), 4);
    for (i, label) in ["5 小时", "1 天", "7 天", "30d"].iter().enumerate() {
        assert_eq!(quotas[i]["label"], json!(format!("{label}周期剩余额度")));
    }
    assert_close(quotas[0]["leftAmount"].as_f64(), 0.0);
}

#[tokio::test]
async fn sub2api_template_reports_invalid_key_and_access_errors() {
    for (status, body, expected) in [
        (200, json!({"isValid": false, "balance": 100}), "Key 不可用"),
        (401, json!({}), "API Key 无效"),
        (403, json!({}), "访问被拒绝"),
        (404, json!({}), "/v1/usage"),
    ] {
        let result = run_builtin_template("SUB2API", status, body, json!({})).await;
        assert_eq!(result.result.status, "error");
        assert!(result.result.payload.unwrap()["message"]
            .as_str()
            .unwrap()
            .contains(expected));
    }
}

#[tokio::test]
async fn kimi_template_accepts_either_single_valid_window() {
    for (usages, label, used) in [
        (json!({"limit_5h": {"used_ratio": 0}}), "5 小时", 0.0),
        (json!({"limit_7d": {"used_ratio": "0.375"}}), "Weekly", 37.5),
        (
            json!({"limit_5h": {"used_ratio": "invalid"}, "limit_7d": {"used_ratio": 0}}),
            "Weekly",
            0.0,
        ),
        (
            json!({"limit_5h": {"used_ratio": 0.5}, "limit_7d": {"used_ratio": -1}}),
            "5 小时",
            50.0,
        ),
    ] {
        let result =
            run_builtin_template("KIMI_CODING", 200, json!({"usages": usages}), json!({})).await;
        assert_eq!(result.result.status, "ok");
        assert_eq!(result.result.error, None);
        let payload = result.result.payload.unwrap();
        let quotas = payload["quotas"].as_array().unwrap();
        assert_eq!(quotas.len(), 1);
        assert_eq!(quotas[0]["label"], json!(label));
        assert_close(quotas[0]["usedPercent"].as_f64(), used);
        assert_close(quotas[0]["leftPercent"].as_f64(), 100.0 - used);
    }
}

#[tokio::test]
async fn kimi_template_rejects_two_invalid_windows() {
    for value in [
        json!(-1),
        json!(true),
        json!("nan"),
        json!("inf"),
        json!("1e309"),
        Value::Null,
    ] {
        let result = run_builtin_template(
            "KIMI_CODING",
            200,
            json!({"usages": {
                "limit_5h": {"used_ratio": value.clone()},
                "limit_7d": {"used_ratio": value}
            }}),
            json!({}),
        )
        .await;
        assert_eq!(result.result.status, "error");
        assert_eq!(
            result.result.error.as_deref(),
            Some("响应缺少有效的 usages.limit_5h / limit_7d.used_ratio")
        );
    }
}

#[tokio::test]
async fn malformed_script_returns_are_errors_in_preview_and_persisted_results() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let eng = engine(&db, None);
    for (index, value) in [
        "nil",
        "false",
        "42",
        "\"not an object\"",
        "{1, 2}",
        "{ summary = \"missing status\" }",
        "{ status = false }",
        "{ status = \"unknown\" }",
    ]
    .iter()
    .enumerate()
    {
        let name = format!("invalid-{index}");
        plugin(
            &db,
            &name,
            &format!("MB = {{}}\nfunction MB.query(ctx) return {value} end"),
        );
        let p = bound_provider(&name, &name, &[("https://a", "sk-t")]);
        let preview = eng.test_provider(&p).await;
        assert_eq!(only(&preview).result.status, "error", "{value}");
        assert!(
            only(&preview)
                .result
                .error
                .as_deref()
                .is_some_and(|error| !error.is_empty()),
            "{value}"
        );
        assert!(db.list_quota_results(&p.key).unwrap().is_empty());
        let persisted = eng.run_provider(&p).await.unwrap();
        assert_eq!(only(&persisted).result.status, "error", "{value}");
        assert_eq!(only(&persisted).result.error, only(&preview).result.error);
        assert_eq!(stored(&db, &p.key, 0).result.status, "error");
    }
}

#[tokio::test]
async fn zero_amounts_remain_numeric_zero_without_percent_fields() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    plugin(
        &db,
        "quota-zero",
        r#"MB = {}
function MB.query(ctx)
  return { status = "ok", quotas = {{ type = "quota", label = "余额", unit = "$", used_amount = 0, left_amount = 0 }} }
end"#,
    );
    let p = bound_provider("zero", "quota-zero", &[("https://a", "sk-t")]);
    let results = engine(&db, None).run_provider(&p).await.unwrap();
    assert_eq!(only(&results).result.status, "ok");
    for payload in [
        only(&results).result.payload.clone().unwrap(),
        stored(&db, "zero", 0).result.payload.unwrap(),
    ] {
        let quota = &payload["quotas"][0];
        assert_close(quota["usedAmount"].as_f64(), 0.0);
        assert_close(quota["leftAmount"].as_f64(), 0.0);
        assert!(quota.get("usedPercent").is_none());
        assert!(quota.get("leftPercent").is_none());
    }
}

#[tokio::test]
async fn default_engine_blocks_local_requests_without_leaking_the_token() {
    let origin = spawn_mock_quota().await;
    let db = Arc::new(Database::open_in_memory().unwrap());
    plugin(
        &db,
        "quota-blocked",
        r#"MB = {}
function MB.query(ctx)
  mb.http.request({ method = "POST", url = ctx.base_url .. "/quota?token=" .. ctx.key,
    headers = {{ "Authorization", "Bearer " .. ctx.key }} })
  return { status = "ok" }
end"#,
    );
    let p = bound_provider(
        "blocked",
        "quota-blocked",
        &[(&origin, "secret-token-never-in-errors")],
    );
    let results = engine(&db, None).test_provider(&p).await;
    let result = &only(&results).result;
    assert_eq!(result.status, "error");
    let error = result.error.as_deref().unwrap();
    assert_eq!(error, "配额 HTTP 请求禁止访问私网或保留地址");
    assert!(!error.contains("secret-token-never-in-errors"));
    assert_eq!(result.payload, None);
}

#[tokio::test]
async fn proxy_policy_errors_are_explicit_but_lua_errors_stay_redacted() {
    let proxy = "http://proxy-private-secret@proxy.example:8080";
    let expected = "配额查询不支持出站代理：无法安全校验代理侧 DNS";
    let error = QuotaNetworkPolicy::from_config(vec![], Some(proxy.into())).unwrap_err();
    assert_eq!(error, expected);
    assert!(!error.contains("proxy-private-secret"));

    let db = Arc::new(Database::open_in_memory().unwrap());
    let eng = engine(&db, None)
        .with_network_policy(QuotaNetworkPolicy::from_environment(Some(proxy.into())));
    plugin(
        &db,
        "quota-proxy",
        r#"MB = {}
function MB.query(ctx)
  mb.http.request({ method = "GET", url = ctx.base_url .. "/quota?token=" .. ctx.key,
    headers = {{ "Authorization", "Bearer " .. ctx.key }} })
  return { status = "ok" }
end"#,
    );
    let p = bound_provider(
        "proxy-denied",
        "quota-proxy",
        &[("https://a", "private-api-key-never-in-errors")],
    );
    let results = eng.test_provider(&p).await;
    let result = &only(&results).result;
    assert_eq!(result.status, "error");
    assert_eq!(result.error.as_deref(), Some(expected));
    assert!(!result
        .error
        .as_deref()
        .unwrap()
        .contains("private-api-key-never-in-errors"));
    assert!(!result
        .error
        .as_deref()
        .unwrap()
        .contains("proxy-private-secret"));
    assert_eq!(result.payload, None);
    assert!(db.list_quota_results(&p.key).unwrap().is_empty());

    for (script, expected_error) in [
        ("error('secret')", "配额脚本加载失败"),
        (
            "MB = {}\nfunction MB.query(ctx) error('secret:' .. ctx.key) end",
            "配额脚本执行失败",
        ),
    ] {
        plugin(&db, "quota-proxy", script);
        let results = eng.test_provider(&p).await;
        let result = &only(&results).result;
        assert_eq!(result.status, "error");
        assert_eq!(result.error.as_deref(), Some(expected_error));
        assert!(!result.error.as_deref().unwrap().contains("secret"));
        assert!(!result
            .error
            .as_deref()
            .unwrap()
            .contains("private-api-key-never-in-errors"));
        assert_eq!(result.payload, None);
        assert!(db.list_quota_results(&p.key).unwrap().is_empty());
    }
}
