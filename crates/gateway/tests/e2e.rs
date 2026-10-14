//! 端到端集成测试：多协议入口 → mock 多协议上游。
//!
//! 覆盖 M6 全矩阵链路：起本地 mock 上游（Anthropic / OpenAI Chat / Gemini），经真实
//! `dispatch::handle_request` 走完整生命周期（路由解析 → Core IR 转换 → 上游调用
//! → 回程转换 → usage 写入数据库），断言客户端最终收到的报文。入口协议覆盖
//! Responses / Anthropic / Chat，上游覆盖 Anthropic / Chat / Gemini。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use axum::body::Body;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use moonbridge_core::Protocol;
use moonbridge_gateway::{bootstrap, dispatch, server, AppState, GatewayConfig};
use moonbridge_store::{
    Database, Endpoint, ModelDef, Offer, PluginRecord, Provider, Route, UsageQuery,
};
use serde_json::{json, Value};

/// 起一个 mock Anthropic 上游（`POST /v1/messages`），按请求 `stream` 返回 JSON 或 SSE。
async fn spawn_mock_anthropic() -> String {
    async fn messages(Json(body): Json<Value>) -> Response {
        if body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            // 标准 Anthropic Messages SSE 事件序列
            let sse = concat!(
                "event: message_start\n",
                "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
                "event: content_block_start\n",
                "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n",
                "event: content_block_stop\n",
                "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "event: message_delta\n",
                "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
                "event: message_stop\n",
                "data: {\"type\":\"message_stop\"}\n\n",
            );
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from(sse))
                .unwrap()
        } else {
            Json(json!({
                "id": "msg_1",
                "model": "claude-x",
                "role": "assistant",
                "content": [{ "type": "text", "text": "Hello from mock" }],
                "stop_reason": "end_turn",
                "usage": { "input_tokens": 10, "output_tokens": 5 }
            }))
            .into_response()
        }
    }

    let app = Router::new().route("/v1/messages", post(messages));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// 构造指向 mock 上游的网关状态（provider `mock` + route `test-model`）。
async fn setup_state() -> (Arc<AppState>, Arc<Database>) {
    let base_url = spawn_mock_anthropic().await;
    let db = Arc::new(Database::open_in_memory().unwrap());
    db.upsert_provider(&Provider {
        key: "mock".into(),
        endpoints: vec![Endpoint {
            protocol: "anthropic".into(),
            base_url,
            api_key: "sk-test".into(),
        }],
        version: Some("2023-06-01".into()),
        user_agent: None,
        web_search: None,
        extra: json!({}),
        quota_plugin_ref: String::new(),
        quota_interval_secs: 0,
        quota_enabled: false,
        quota_config: serde_json::json!({}),
        enabled: true,
        created_at: 0,
        updated_at: 0,
    })
    .unwrap();
    db.upsert_route(&Route {
        alias: "test-model".into(),
        model_slug: "claude-x".into(),
        provider_key: "mock".into(),
        extra: Value::Null,
    })
    .unwrap();
    // 计价数据流：offer 定价存在时 usage 记录的 cost 应按价目实时计算。
    db.upsert_offer(&Offer {
        provider_key: "mock".into(),
        model_slug: "claude-x".into(),
        pricing: Some(json!({ "input": 3.0, "output": 15.0 })),
        endpoint_protocol: None,
    })
    .unwrap();
    let state = bootstrap(GatewayConfig::default(), db.clone()).unwrap();
    (state, db)
}

/// 非流式：Responses 入口 → Anthropic 上游 → Responses 输出。
#[tokio::test]
async fn e2e_non_stream_responses_to_anthropic() {
    let (state, db) = setup_state().await;

    let body = json!({
        "model": "test-model",
        "instructions": "You are helpful",
        "input": "Hi",
        "stream": false
    });
    let resp = dispatch::handle_request(state, Protocol::OpenAiResponse, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    assert_eq!(resp.status(), 200);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let out: Value = serde_json::from_slice(&bytes).unwrap();

    // 客户端收到的是 OpenAI Responses 对象
    assert_eq!(out["object"], "response");
    assert_eq!(out["status"], "completed");
    assert!(out["output"].is_array(), "应含 output 数组: {out}");
    assert!(
        out.to_string().contains("Hello from mock"),
        "上游文本应流转到 Responses 输出: {out}"
    );
    assert_eq!(out["usage"]["input_tokens"], 10);

    // usage 已写入数据库
    let sum = db.usage_summary().unwrap();
    assert_eq!(sum.requests, 1);
    assert_eq!(sum.input_tokens, 10);
    assert_eq!(sum.output_tokens, 5);

    // 成本按命中 offer 的价目实时计算：(10 in × $3 + 5 out × $15) / 1M
    let rows = db
        .query_usage(&UsageQuery {
            limit: 1,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rows[0].provider_key.as_deref(), Some("mock"));
    assert!(
        (rows[0].cost - 0.000105).abs() < 1e-9,
        "cost 应为 (10×3 + 5×15)/1e6 = 0.000105，实际 {}",
        rows[0].cost
    );
}

/// 流式：Responses 入口（stream=true）→ Anthropic SSE → Responses SSE。
#[tokio::test]
async fn e2e_stream_responses_to_anthropic() {
    let (state, db) = setup_state().await;

    let body = json!({
        "model": "test-model",
        "input": "Hi",
        "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::OpenAiResponse, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    assert_eq!(resp.status(), 200);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes);

    // 客户端收到 SSE，且上游文本增量完整流转
    assert!(text.contains("data:"), "应为 SSE 格式: {text}");
    assert!(text.contains("Hello"), "SSE 应含首段增量: {text}");
    assert!(text.contains("world"), "SSE 应含次段增量: {text}");

    // 流式请求同样写入数据库一条 usage
    let sum = db.usage_summary().unwrap();
    assert_eq!(sum.requests, 1);

    // 收尾状态由 `StreamAudit` 的 Drop 落笔：断言它是 "ok" 同时证明
    // 「生成器内部对 audit 字段的写入能被 Drop 读到」（rustc 的
    // unused_assignments 只看字段级数据流，不认 Drop glue，会误报）。
    let rows = db
        .query_usage(&UsageQuery {
            limit: 5,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        rows[0].status.as_deref(),
        Some("ok"),
        "正常读尽的流应记 ok，实际 {:?}",
        rows[0].status
    );
    assert!(rows[0].ttft_ms.is_some(), "流式请求必须记录首字延迟 TTFT");
    assert_eq!(
        rows[0].input_tokens, 10,
        "流式 usage 应从 MessageDelta 累积"
    );
}

/// 起一个返回固定 SSE 正文的 mock Anthropic 上游（流式路径专用：正文即整个响应体）。
async fn spawn_mock_sse_body(sse: &'static str) -> String {
    let app = Router::new().route(
        "/v1/messages",
        post(move || async move {
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from(sse))
                .unwrap()
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

async fn collect_stream(state: Arc<AppState>) -> String {
    let body = json!({
        "model": "test-model",
        "max_tokens": 128,
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    assert_eq!(resp.status(), 200);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8_lossy(&bytes).to_string()
}

/// 回归（NewAPI 中转断流）：上游 SSE 有内容但干净 EOF、从未发终帧。
/// 历史缺陷是把截断伪造成正常完成记 ok；现在必须向下游发带内 error
/// 事件、保留已产出的部分内容，并把请求记为 error。
#[tokio::test]
async fn e2e_stream_truncated_eof_reports_error() {
    let sse = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n",
        // 无 message_delta / message_stop——连接干净 EOF（中转断流的典型形态）
    );
    let (state, db) = setup_with(
        "anthropic",
        spawn_mock_sse_body(sse).await,
        "claude-x",
        "test-model",
    )
    .await;
    let text = collect_stream(state).await;

    assert!(text.contains("partial"), "已产出内容应保留: {text}");
    assert!(
        text.contains("\"type\":\"error\""),
        "截断流应发带内 error 事件: {text}"
    );
    let rows = db
        .query_usage(&UsageQuery {
            limit: 5,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].status.as_deref(),
        Some("error"),
        "缺终帧的截断流不得记 ok"
    );
    assert!(
        rows[0].error.as_deref().unwrap_or("").contains("终止帧"),
        "错误应说明缺终止帧: {:?}",
        rows[0].error
    );
}

/// 回归：上游 200 + 零事件空流（NewAPI 空转的实测形态）——不得伪造消息壳，
/// 发 Error + 终结帧，记 error。
#[tokio::test]
async fn e2e_stream_empty_eof_reports_error() {
    let (state, db) = setup_with(
        "anthropic",
        spawn_mock_sse_body("").await,
        "claude-x",
        "test-model",
    )
    .await;
    let text = collect_stream(state).await;

    assert!(
        text.contains("\"type\":\"error\""),
        "空流应发带内 error 事件: {text}"
    );
    assert!(
        !text.contains("message_start"),
        "空流不得伪造消息壳: {text}"
    );
    let rows = db
        .query_usage(&UsageQuery {
            limit: 5,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rows[0].status.as_deref(), Some("error"), "空流不得记 ok");
}

/// 上游主动发错误帧（anthropic error 事件）后 EOF：上游自己宣告失败，
/// 请求必须记 error——错误帧后即使缺终帧也不能记 ok。
#[tokio::test]
async fn e2e_stream_upstream_error_frame_records_error() {
    let sse = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
        "event: error\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"upstream overloaded\"}}\n\n",
        // 上游发 error 后干净 EOF，无 message_delta / message_stop
    );
    let (state, db) = setup_with(
        "anthropic",
        spawn_mock_sse_body(sse).await,
        "claude-x",
        "test-model",
    )
    .await;
    let text = collect_stream(state).await;

    assert!(text.contains("\"type\":\"error\""), "错误帧应透传: {text}");
    let rows = db
        .query_usage(&UsageQuery {
            limit: 5,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(
        rows[0].status.as_deref(),
        Some("error"),
        "上游错误帧不得记 ok"
    );
}

// ============================================================================
// M6 全矩阵：更多 mock 上游 + 通用装配 + 跨协议链路
// ============================================================================

/// 起一个 mock OpenAI Chat 上游（`POST /v1/chat/completions`）。
async fn spawn_mock_openai_chat() -> String {
    async fn chat(Json(body): Json<Value>) -> Response {
        if body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            let sse = concat!(
                "data: {\"id\":\"c1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
                "data: {\"id\":\"c1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n",
                "data: {\"id\":\"c1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":4,\"total_tokens\":12}}\n\n",
                "data: [DONE]\n\n",
            );
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from(sse))
                .unwrap()
        } else {
            Json(json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [{ "index": 0, "message": { "role": "assistant", "content": "Hello from chat" }, "finish_reason": "stop" }],
                "usage": { "prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12 }
            }))
            .into_response()
        }
    }
    let app = Router::new().route("/v1/chat/completions", post(chat));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// 起一个 mock Gemini 上游。Gemini 路径含模型名与 `:generateContent` / `:streamGenerateContent`
/// 后缀，用 fallback 捕获全部路径，按路径是否含 `streamGenerateContent` 决定流式与否。
async fn spawn_mock_gemini() -> String {
    async fn handler(uri: axum::http::Uri, Json(_body): Json<Value>) -> Response {
        if uri.path().contains("streamGenerateContent") {
            let sse = concat!(
                "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello\"}]},\"index\":0}],\"modelVersion\":\"gemini-2.0-flash\",\"responseId\":\"r1\"}\n\n",
                "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\" world\"}]},\"index\":0}]}\n\n",
                "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"!\"}]},\"finishReason\":\"STOP\",\"index\":0}],\"usageMetadata\":{\"promptTokenCount\":8,\"candidatesTokenCount\":4,\"totalTokenCount\":12}}\n\n",
            );
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from(sse))
                .unwrap()
        } else {
            Json(json!({
                "candidates": [{ "content": { "role": "model", "parts": [{ "text": "Hello from gemini" }] }, "finishReason": "STOP", "index": 0 }],
                "usageMetadata": { "promptTokenCount": 8, "candidatesTokenCount": 4, "totalTokenCount": 12 },
                "modelVersion": "gemini-2.0-flash",
                "responseId": "resp-1"
            }))
            .into_response()
        }
    }
    let app = Router::new().fallback(handler);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// 通用装配：以指定上游协议 + base_url 构造 provider `mock` 与 route `alias → model_slug`。
async fn setup_with(
    protocol: &str,
    base_url: String,
    upstream_model: &str,
    alias: &str,
) -> (Arc<AppState>, Arc<Database>) {
    let db = Arc::new(Database::open_in_memory().unwrap());
    db.upsert_provider(&Provider {
        key: "mock".into(),
        endpoints: vec![Endpoint {
            protocol: protocol.into(),
            base_url,
            api_key: "sk-test".into(),
        }],
        version: None,
        user_agent: None,
        web_search: None,
        extra: json!({}),
        quota_plugin_ref: String::new(),
        quota_interval_secs: 0,
        quota_enabled: false,
        quota_config: serde_json::json!({}),
        enabled: true,
        created_at: 0,
        updated_at: 0,
    })
    .unwrap();
    db.upsert_route(&Route {
        alias: alias.into(),
        model_slug: upstream_model.into(),
        provider_key: "mock".into(),
        extra: Value::Null,
    })
    .unwrap();
    let state = bootstrap(GatewayConfig::default(), db.clone()).unwrap();
    (state, db)
}

/// Chat 入口 → Chat 上游（非流式）。
#[tokio::test]
async fn e2e_non_stream_chat_to_chat() {
    let base_url = spawn_mock_openai_chat().await;
    let (state, db) = setup_with("openai-chat", base_url, "gpt-4o", "test-model").await;

    let body = json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let resp = dispatch::handle_request(state, Protocol::OpenAiChat, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    assert_eq!(resp.status(), 200);

    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let out: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(
        untag(out["choices"][0]["message"]["content"].as_str().unwrap()),
        "Hello from chat"
    );
    assert_eq!(out["usage"]["prompt_tokens"], 8);

    let sum = db.usage_summary().unwrap();
    assert_eq!(sum.requests, 1);
    assert_eq!(sum.input_tokens, 8);
}

/// Chat 入口 → Chat 上游（流式 SSE）。
#[tokio::test]
async fn e2e_stream_chat_to_chat() {
    let base_url = spawn_mock_openai_chat().await;
    let (state, db) = setup_with("openai-chat", base_url, "gpt-4o", "test-model").await;

    let body = json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::OpenAiChat, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("Hello"), "SSE 应含首段增量: {text}");
    assert!(text.contains("world"), "SSE 应含次段增量: {text}");
    assert!(text.contains("[DONE]"), "Chat SSE 应以 [DONE] 结束: {text}");
    assert_eq!(db.usage_summary().unwrap().requests, 1);
}

/// 记录上游请求体、并返回「thinking 模式」流式响应（reasoning_content +
/// tool_calls）的 mock OpenAI Chat 上游。返回 (base_url, 捕获槽)。
async fn spawn_mock_chat_thinking_capture() -> (String, Arc<std::sync::Mutex<Vec<Value>>>) {
    let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
    let cap = captured.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move |Json(body): Json<Value>| {
            let cap = cap.clone();
            async move {
                cap.lock().unwrap().push(body);
                let sse = concat!(
                    "data: {\"id\":\"c1\",\"model\":\"ds\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"think hard\"}}]}\n\n",
                    "data: {\"id\":\"c1\",\"model\":\"ds\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"Bash\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}}]}}]}\n\n",
                    "data: {\"id\":\"c1\",\"model\":\"ds\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20}}\n\n",
                    "data: [DONE]\n\n",
                );
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from(sse))
                    .unwrap()
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), captured)
}

/// 端到端回归：Responses 入口 → Chat thinking 上游的 reasoning_content
/// 必须经客户端历史回传到上游——
///   1. 下发给客户端的 reasoning item 必须带 encrypted_content（Codex 系
///      客户端只回传带凭据的 reasoning item）；
///   2. 客户端回传 reasoning item 后，出站 assistant+tool_calls 消息必须
///      携带 reasoning_content（上游 thinking 校验缺失即 400）。
#[tokio::test]
async fn e2e_reasoning_content_roundtrips_responses_to_chat() {
    let (base_url, captured) = spawn_mock_chat_thinking_capture().await;
    let (state, _db) = setup_with("openai-chat", base_url, "deepseek-x", "test-model").await;

    // 第 1 轮：客户端只发 user，上游返回 reasoning + tool_calls
    let body = json!({
        "model": "test-model",
        "stream": true,
        "input": [{ "role": "user", "content": [{ "type": "input_text", "text": "run ls" }] }]
    });
    let resp =
        dispatch::handle_request(state.clone(), Protocol::OpenAiResponse, body, vec![], None)
            .await
            .expect("dispatch 应成功");
    assert_eq!(resp.status(), 200);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let sse = String::from_utf8_lossy(&bytes);
    assert!(
        sse.contains("\"type\":\"reasoning\"") && sse.contains("encrypted_content"),
        "reasoning item 必须带 encrypted_content 客户端才会回传: {sse}"
    );

    // 第 2 轮：客户端原样回传 reasoning item（含 encrypted_content）+
    // function_call 链——模拟 Codex 系客户端的历史重放
    let body = json!({
        "model": "test-model",
        "stream": true,
        "input": [
            { "role": "user", "content": [{ "type": "input_text", "text": "run ls" }] },
            { "type": "reasoning", "id": "rs_1",
              "summary": [{ "type": "summary_text", "text": "think hard" }],
              "encrypted_content": "chat:think hard" },
            { "type": "function_call", "call_id": "call_1", "name": "Bash",
              "arguments": "{\"command\":\"ls\"}" },
            { "type": "function_call_output", "call_id": "call_1", "output": "a.txt" }
        ]
    });
    let resp = dispatch::handle_request(state, Protocol::OpenAiResponse, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    assert_eq!(resp.status(), 200);

    let reqs = captured.lock().unwrap();
    assert_eq!(reqs.len(), 2);
    let msgs = reqs[1]["messages"].as_array().unwrap();
    let assistant = msgs
        .iter()
        .find(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
        .expect("应有携带 tool_calls 的 assistant 消息");
    assert_eq!(
        assistant["reasoning_content"], "think hard",
        "tool_calls 消息必须携带 reasoning_content（thinking 校验）: {msgs:?}"
    );
}

/// 起一个对任意路径/方法都返回指定状态码的 mock 上游（用于故障转移测试）。
async fn spawn_mock_status(status: axum::http::StatusCode) -> String {
    async fn fail(status: axum::http::StatusCode) -> Response {
        (status, "boom").into_response()
    }
    // fallback：不限路径，保证跨协议 Adapter 的任意请求路径都命中失败响应
    let app = Router::new().fallback(move || fail(status));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// Anthropic 入口 → Anthropic 上游（非流式，同协议直通）。
#[tokio::test]
async fn e2e_non_stream_anthropic_entry() {
    let (state, _db) = setup_with(
        "anthropic",
        spawn_mock_anthropic().await,
        "claude-x",
        "test-model",
    )
    .await;

    let body = json!({
        "model": "test-model",
        "max_tokens": 128,
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let out: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(out["type"], "message");
    assert_eq!(out["role"], "assistant");
    assert_eq!(
        untag(out["content"][0]["text"].as_str().unwrap()),
        "Hello from mock"
    );
    assert_eq!(out["usage"]["input_tokens"], 10);
}

/// Chat 入口 → Gemini 上游（非流式，跨协议）。
#[tokio::test]
async fn e2e_non_stream_chat_to_gemini() {
    let base_url = spawn_mock_gemini().await;
    let (state, db) = setup_with("google-genai", base_url, "gemini-2.0-flash", "test-model").await;

    let body = json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let resp = dispatch::handle_request(state, Protocol::OpenAiChat, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let out: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(
        untag(out["choices"][0]["message"]["content"].as_str().unwrap()),
        "Hello from gemini"
    );
    assert_eq!(db.usage_summary().unwrap().input_tokens, 8);
}

/// Anthropic 入口 → Gemini 上游（流式，跨协议 SSE 转换）。
#[tokio::test]
async fn e2e_stream_anthropic_entry_to_gemini() {
    let base_url = spawn_mock_gemini().await;
    let (state, db) = setup_with("google-genai", base_url, "gemini-2.0-flash", "test-model").await;

    let body = json!({
        "model": "test-model",
        "max_tokens": 128,
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes);
    // Anthropic 入口 SSE：message_start → content_block_delta(text) → message_stop
    assert!(
        text.contains("message_start"),
        "应含 Anthropic 起始事件: {text}"
    );
    assert!(text.contains("Hello"), "应含首段增量: {text}");
    assert!(text.contains("world"), "应含次段增量: {text}");
    assert!(
        text.contains("message_stop"),
        "应含 Anthropic 结束事件: {text}"
    );
    assert_eq!(db.usage_summary().unwrap().requests, 1);
}

/// 多端点故障转移：连接拒绝 + 500 + 429 的端点依次跳过，末位健康端点成功。
#[tokio::test]
async fn e2e_failover_to_healthy_endpoint() {
    let good = spawn_mock_anthropic().await;
    let server_500 = spawn_mock_status(axum::http::StatusCode::INTERNAL_SERVER_ERROR).await;
    let server_429 = spawn_mock_status(axum::http::StatusCode::TOO_MANY_REQUESTS).await;

    let db = Arc::new(Database::open_in_memory().unwrap());
    db.upsert_provider(&Provider {
        key: "mock".into(),
        endpoints: vec![
            // 连接拒绝（未监听端口）
            Endpoint {
                protocol: "anthropic".into(),
                base_url: "http://127.0.0.1:9".into(),
                api_key: "sk-first".into(),
            },
            Endpoint {
                protocol: "anthropic".into(),
                base_url: server_500,
                api_key: "sk-second".into(),
            },
            Endpoint {
                protocol: "anthropic".into(),
                base_url: server_429,
                api_key: "".into(),
            }, // 空 Key 沿用上一非空
            Endpoint {
                protocol: "anthropic".into(),
                base_url: good,
                api_key: "sk-good".into(),
            },
        ],
        version: Some("2023-06-01".into()),
        user_agent: None,
        web_search: None,
        extra: json!({}),
        quota_plugin_ref: String::new(),
        quota_interval_secs: 0,
        quota_enabled: false,
        quota_config: serde_json::json!({}),
        enabled: true,
        created_at: 0,
        updated_at: 0,
    })
    .unwrap();
    db.upsert_route(&Route {
        alias: "test-model".into(),
        model_slug: "claude-x".into(),
        provider_key: "mock".into(),
        extra: Value::Null,
    })
    .unwrap();
    let state = bootstrap(GatewayConfig::default(), db.clone()).unwrap();

    let body = json!({
        "model": "test-model",
        "max_tokens": 128,
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("故障转移后应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let out: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        untag(out["content"][0]["text"].as_str().unwrap()),
        "Hello from mock"
    );
    assert_eq!(db.usage_summary().unwrap().requests, 1);
}

/// 非 4xx/5xx 可重试类错误：4xx 快速失败，不转移。
#[tokio::test]
async fn e2e_no_failover_on_client_error() {
    let good = spawn_mock_anthropic().await;
    let bad = spawn_mock_status(axum::http::StatusCode::UNAUTHORIZED).await;

    let db = Arc::new(Database::open_in_memory().unwrap());
    db.upsert_provider(&Provider {
        key: "mock".into(),
        endpoints: vec![
            Endpoint {
                protocol: "anthropic".into(),
                base_url: bad,
                api_key: "sk-test".into(),
            },
            Endpoint {
                protocol: "anthropic".into(),
                base_url: good,
                api_key: "sk-test".into(),
            },
        ],
        version: Some("2023-06-01".into()),
        user_agent: None,
        web_search: None,
        extra: json!({}),
        quota_plugin_ref: String::new(),
        quota_interval_secs: 0,
        quota_enabled: false,
        quota_config: serde_json::json!({}),
        enabled: true,
        created_at: 0,
        updated_at: 0,
    })
    .unwrap();
    db.upsert_route(&Route {
        alias: "test-model".into(),
        model_slug: "claude-x".into(),
        provider_key: "mock".into(),
        extra: Value::Null,
    })
    .unwrap();
    let state = bootstrap(GatewayConfig::default(), db.clone()).unwrap();

    let body = json!({
        "model": "test-model",
        "max_tokens": 128,
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let err = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect_err("401 应直接失败");
    assert!(matches!(
        err,
        moonbridge_gateway::GatewayError::Upstream { status: 401, .. }
    ));
    // 仅首端点被尝试（错误请求也落一条 usage，但未发生转移）
    assert_eq!(db.usage_summary().unwrap().requests, 1);
}

/// 跨协议故障转移：首个端点（openai-chat）500 后切换到 Anthropic 端点成功，
/// 验证协议绑定在端点上时 Adapter 逐端点选取。
#[tokio::test]
async fn e2e_failover_across_protocols() {
    let anthropic = spawn_mock_anthropic().await;
    let chat_500 = spawn_mock_status(axum::http::StatusCode::INTERNAL_SERVER_ERROR).await;

    let db = Arc::new(Database::open_in_memory().unwrap());
    db.upsert_provider(&Provider {
        key: "mock".into(),
        endpoints: vec![
            Endpoint {
                protocol: "openai-chat".into(),
                base_url: chat_500,
                api_key: "sk-a".into(),
            },
            Endpoint {
                protocol: "anthropic".into(),
                base_url: anthropic,
                api_key: "sk-b".into(),
            },
        ],
        version: None,
        user_agent: None,
        web_search: None,
        extra: json!({}),
        quota_plugin_ref: String::new(),
        quota_interval_secs: 0,
        quota_enabled: false,
        quota_config: serde_json::json!({}),
        enabled: true,
        created_at: 0,
        updated_at: 0,
    })
    .unwrap();
    db.upsert_route(&Route {
        alias: "test-model".into(),
        model_slug: "claude-x".into(),
        provider_key: "mock".into(),
        extra: Value::Null,
    })
    .unwrap();
    let state = bootstrap(GatewayConfig::default(), db.clone()).unwrap();

    // Anthropic 入口；首个端点按 openai-chat 构造请求失败后转移到 Anthropic 端点
    let body = json!({
        "model": "test-model",
        "max_tokens": 128,
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("跨协议转移后应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let out: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        untag(out["content"][0]["text"].as_str().unwrap()),
        "Hello from mock"
    );
    assert_eq!(db.usage_summary().unwrap().requests, 1);
}

// ── 钩子接线与短路审计 ───────────────────────────────────────────────

/// 无论收到什么请求都返回固定报文的 mock Anthropic 上游。
async fn spawn_mock_returning(body: Value) -> String {
    let app = Router::new()
        .route(
            "/v1/messages",
            post(|axum::extract::State(b): axum::extract::State<Value>| async move { Json(b) }),
        )
        .with_state(body);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// 记录上游请求体的 mock Anthropic（断言网关出站报文用）。返回 (base_url, 捕获槽)。
async fn spawn_mock_anthropic_capture() -> (String, Arc<std::sync::Mutex<Vec<Value>>>) {
    let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
    let cap = captured.clone();
    let app = Router::new().route(
        "/v1/messages",
        post(move |Json(body): Json<Value>| {
            let cap = cap.clone();
            async move {
                cap.lock().unwrap().push(body);
                Json(json!({
                    "id": "msg_1", "model": "claude-x", "role": "assistant",
                    "content": [{ "type": "text", "text": "ok" }],
                    "stop_reason": "end_turn",
                    "usage": { "input_tokens": 1, "output_tokens": 1 }
                }))
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), captured)
}

/// 回归：客户端未设输出上限时，Anthropic 上游的必填 `max_tokens` 按模型元数据
/// （`models.max_output_tokens`）回退——不得凭空注入小值截断输出
/// （线上实证 OpenCode Zen 端被静默注入 4096、输出恰好截在 4096）。
#[tokio::test]
async fn e2e_anthropic_max_tokens_uses_model_output_limit() {
    let (base_url, captured) = spawn_mock_anthropic_capture().await;
    let (state, db) = setup_with("anthropic", base_url, "claude-x", "test-model").await;
    db.upsert_model(&ModelDef {
        slug: "claude-x".into(),
        display_name: None,
        context_window: Some(200_000),
        max_output_tokens: Some(131_072),
        modalities: None,
        reasoning_levels: None,
        extra: Value::Null,
    })
    .unwrap();

    // 客户端不设 max_tokens
    let body = json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "Hi" }],
    });
    let resp = dispatch::handle_request(state.clone(), Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    assert_eq!(resp.status(), 200);
    {
        let reqs = captured.lock().unwrap();
        assert_eq!(reqs.len(), 1);
        assert_eq!(
            reqs[0]["max_tokens"], 131_072,
            "客户端未设上限时应按模型输出上限回退: {}",
            reqs[0]
        );
    }

    // 无模型元数据时回退到常量（必填字段仍须发送）
    db.delete_model("claude-x").unwrap();
    let body = json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "Hi" }],
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    assert_eq!(resp.status(), 200);
    let reqs = captured.lock().unwrap();
    assert_eq!(reqs.len(), 2);
    assert_eq!(
        reqs[1]["max_tokens"], 4096,
        "无元数据时回退到常量: {}",
        reqs[1]
    );
}

/// 恒返回同一段 SSE 的 mock Anthropic 上游（流式测试用）。
async fn spawn_mock_sse(sse: String) -> String {
    let app = Router::new()
        .route(
            "/v1/messages",
            post(
                |axum::extract::State(b): axum::extract::State<String>| async move {
                    Response::builder()
                        .header("content-type", "text/event-stream")
                        .body(Body::from(b))
                        .unwrap()
                },
            ),
        )
        .with_state(sse);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{addr}")
}

/// 指向 `base_url` 的内联 Lua 插件网关状态 + 独立 trace 目录。
async fn seed_lua_state(
    tag: &str,
    plugin_script: &str,
    capabilities: &[&str],
    base_url: String,
) -> (Arc<AppState>, Arc<Database>, std::path::PathBuf) {
    let db = Arc::new(Database::open_in_memory().unwrap());
    db.upsert_provider(&Provider {
        key: "mock".into(),
        endpoints: vec![Endpoint {
            protocol: "anthropic".into(),
            base_url,
            api_key: "sk-test".into(),
        }],
        version: Some("2023-06-01".into()),
        user_agent: None,
        web_search: None,
        extra: json!({}),
        quota_plugin_ref: String::new(),
        quota_interval_secs: 0,
        quota_enabled: false,
        quota_config: serde_json::json!({}),
        enabled: true,
        created_at: 0,
        updated_at: 0,
    })
    .unwrap();
    db.upsert_route(&Route {
        alias: "test-model".into(),
        model_slug: "claude-x".into(),
        provider_key: "mock".into(),
        extra: Value::Null,
    })
    .unwrap();
    db.upsert_plugin(&PluginRecord {
        name: format!("e2e-{tag}"),
        source: "lua".into(),
        script_ref: plugin_script.into(),
        enabled: true,
        config: json!({}),
        scopes: vec!["global".into()],
        capabilities: capabilities.iter().map(|c| c.to_string()).collect(),
        category: "core".into(),
        config_schema: serde_json::Value::Null,
    })
    .unwrap();

    let trace_dir = std::env::temp_dir().join(format!("mb-e2e-trace-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&trace_dir);
    let cfg = GatewayConfig {
        trace_dir: Some(trace_dir.to_string_lossy().to_string()),
        ..GatewayConfig::default()
    };
    let state = bootstrap(cfg, db.clone()).expect("bootstrap 应成功加载插件");
    (state, db, trace_dir)
}

/// 非流式：内联 Lua 插件 + 固定 JSON mock 上游。
async fn setup_state_lua(
    tag: &str,
    plugin_script: &str,
    capabilities: &[&str],
    mock_body: Value,
) -> (Arc<AppState>, Arc<Database>, std::path::PathBuf) {
    let base_url = spawn_mock_returning(mock_body).await;
    seed_lua_state(tag, plugin_script, capabilities, base_url).await
}

/// 流式：内联 Lua 插件 + 固定 SSE mock 上游。
async fn setup_state_lua_sse(
    tag: &str,
    plugin_script: &str,
    capabilities: &[&str],
    sse: String,
) -> (Arc<AppState>, Arc<Database>, std::path::PathBuf) {
    let base_url = spawn_mock_sse(sse).await;
    seed_lua_state(tag, plugin_script, capabilities, base_url).await
}

/// 递归收集目录下的 `*.json`（trace 文件路径为 `<dir>/<session>/<model>/<ts>-<id>.json`）。
fn trace_json_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().and_then(|x| x.to_str()) == Some("json") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// 剥去尾随的旧式会话水印（形如 ` [mb:xxxxxx]`），返回正文。
///
/// 水印已迁入推理块首部、正文不再有 marker；本函数仅作旧式水印的防御性剥除，
/// 让断言不依赖「响应一定无文本水印」这一配置细节。水印的存在性与稳定性由
/// 专门的 `e2e_session_marker_*` 测试守。
fn untag(text: &str) -> &str {
    match text.rfind(" [mb:") {
        // `" [mb:"` 5 字符 + 6 位 tag + `]` = 12
        Some(i) if text.ends_with(']') && text.len() - i == 12 => &text[..i],
        _ => text,
    }
}

/// 回归：`filter_content` 必须真的被 gateway 调用（此前 trait 与 Lua 实现俱在、
/// 但链路零调用点，插件写 `MB.filter_content` 永不触发）。
#[tokio::test]
async fn e2e_filter_content_drops_blocks() {
    let (state, db, trace_dir) = setup_state_lua(
        "filter",
        r#"
        MB = { version = "0.1.0", capabilities = { "core" } }
        function MB.filter_content(ctx, block)
          return block.type == "tool_use"
        end
        "#,
        &["core"],
        json!({
            "id": "msg_f", "model": "claude-x", "role": "assistant",
            "content": [
                { "type": "text", "text": "keep me" },
                { "type": "tool_use", "id": "t1", "name": "dropme", "input": {} }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 3, "output_tokens": 2 }
        }),
    )
    .await;

    let body = json!({
        "model": "test-model",
        "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes);

    assert!(text.contains("keep me"), "文本块应保留: {text}");
    assert!(
        !text.contains("dropme"),
        "tool_use 块应被 filter_content 丢弃: {text}"
    );
    assert_eq!(db.usage_summary().unwrap().requests, 1);
    assert_eq!(
        trace_json_files(&trace_dir).len(),
        1,
        "正常请求应留下一份 trace"
    );
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 对照组（为上一条测试提供有效性校验）：同样的上游报文，插件声明 `core` 但不实现
/// `filter_content` ⇒ `tool_use` 块必须照常出现。若这条也丢了块，说明丢块与钩子
/// 无关、`e2e_filter_content_drops_blocks` 就形同虚设。
#[tokio::test]
async fn e2e_blocks_survive_without_filter_content() {
    let (state, _db, trace_dir) = setup_state_lua(
        "nofilter",
        r#"
        MB = { version = "0.1.0", capabilities = { "core" } }
        function MB.on_request(ctx, req) return req end
        "#,
        &["core"],
        json!({
            "id": "msg_f", "model": "claude-x", "role": "assistant",
            "content": [
                { "type": "text", "text": "keep me" },
                { "type": "tool_use", "id": "t1", "name": "dropme", "input": {} }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 3, "output_tokens": 2 }
        }),
    )
    .await;

    let body = json!({
        "model": "test-model",
        "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes);

    assert!(
        text.contains("dropme"),
        "无 filter_content 时 tool_use 块应原样到达客户端: {text}"
    );
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 含两个内容块（text + tool_use）的标准 Anthropic SSE 序列。
fn sse_text_then_tool_use() -> String {
    concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":4,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Visible\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"dropme\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":3}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    )
    .to_string()
}

/// 流式对照：无 `filter_content` 时两个块都要原样到达客户端。
#[tokio::test]
async fn e2e_stream_blocks_survive_without_filter_content() {
    let (state, _db, trace_dir) = setup_state_lua_sse(
        "snofilter",
        r#"
        MB = { version = "0.1.0", capabilities = { "core" } }
        function MB.on_request(ctx, req) return req end
        "#,
        &["core"],
        sse_text_then_tool_use(),
    )
    .await;

    let body = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }], "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes).to_string();

    assert!(text.contains("Visible"), "文本块增量应在: {text}");
    assert!(text.contains("dropme"), "tool_use 块应在: {text}");
    assert!(text.contains("partial_json"), "tool_use 增量应在: {text}");
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 回归：流式路径上 `filter_content` 丢块后，该块的 delta / stop 必须一并压制，
/// 否则客户端会收到一个没有 `content_block_start` 的孤立增量。
#[tokio::test]
async fn e2e_stream_filter_content_drops_block_and_its_deltas() {
    let (state, db, trace_dir) = setup_state_lua_sse(
        "sfilter",
        r#"
        MB = { version = "0.1.0", capabilities = { "core" } }
        function MB.filter_content(ctx, block)
          return block.type == "tool_use"
        end
        "#,
        &["core"],
        sse_text_then_tool_use(),
    )
    .await;

    let body = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }], "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes).to_string();

    assert!(text.contains("Visible"), "未命中过滤的文本块应保留: {text}");
    assert!(
        !text.contains("dropme"),
        "tool_use 块的 content_block_start 应被丢弃: {text}"
    );
    assert!(
        !text.contains("partial_json"),
        "被丢块的后续增量必须连带压制: {text}"
    );
    // 流仍应正常收尾并留下用量（丢弃块不得打断链路）
    assert!(text.contains("message_stop"), "流应正常收尾: {text}");
    let rows = db
        .query_usage(&UsageQuery {
            limit: 5,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].status.as_deref(), Some("ok"));
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 回归：`MB.init` / `MB.shutdown` 的扇出必须由 `serve_with_shutdown` **成对**触发
/// （此前整仓零调用点，两个钩子是死的）。用 spy 替身直接验证调用点本身。
#[tokio::test]
async fn e2e_serve_pairs_plugin_init_and_shutdown() {
    use async_trait::async_trait;
    use moonbridge_protocol::{builtin_registry, PluginHooks};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct LifecycleSpy {
        inits: AtomicUsize,
        shutdowns: AtomicUsize,
    }
    #[async_trait]
    impl PluginHooks for LifecycleSpy {
        async fn init_all(&self) {
            self.inits.fetch_add(1, Ordering::SeqCst);
        }
        async fn shutdown_all(&self) {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
        }
    }

    // 让 OS 选端口后释放，再交给 serve 绑定（避免与并行测试撞固定端口）
    let probe = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = probe.local_addr().unwrap();
    drop(probe);

    let spy = Arc::new(LifecycleSpy::default());
    let cfg = GatewayConfig {
        addr: addr.to_string(),
        ..GatewayConfig::default()
    };
    let state = AppState::new(
        cfg,
        Arc::new(Database::open_in_memory().unwrap()),
        Arc::new(builtin_registry()),
        spy.clone(),
        reqwest::Client::new(),
    );

    // 200ms 后给出平滑关闭信号
    server::serve_with_shutdown(state, async {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    })
    .await
    .expect("服务应正常退出");

    assert_eq!(
        spy.inits.load(Ordering::SeqCst),
        1,
        "init_all 必须由 serve 恰好触发一次"
    );
    assert_eq!(
        spy.shutdowns.load(Ordering::SeqCst),
        1,
        "shutdown_all 必须由 serve 恰好触发一次（与 init 配对）"
    );
}

/// 配置 auth_token 后，探活和模型目录保持公开，推理入口要求 Bearer。
#[tokio::test]
async fn e2e_health_and_models_are_public_with_auth_configured() {
    let db = Arc::new(Database::open_in_memory().unwrap());
    let cfg = GatewayConfig {
        auth_token: Some("secret-token".into()),
        ..GatewayConfig::default()
    };
    let state = bootstrap(cfg, db).unwrap();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, server::router(state)).await.unwrap();
    });

    let client = reqwest::Client::new();

    let health = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(health.status(), 200);
    let models_url = format!("http://{addr}/v1/models");
    assert_eq!(client.get(&models_url).send().await.unwrap().status(), 200);
    for authorization in [
        "secret-token",
        "Basic secret-token",
        "Bearer wrong",
        "Bearer ",
    ] {
        let response = client
            .get(&models_url)
            .header("authorization", authorization)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            200,
            "公开模型列表不受鉴权影响：{authorization:?}"
        );
    }
    let models = client
        .get(&models_url)
        .bearer_auth("secret-token")
        .send()
        .await
        .unwrap();
    assert_eq!(models.status(), 200);

    // 受保护面：POST 入口无 token / 错 token 一律 401
    let url = format!("http://{addr}/v1/chat/completions");
    let body = json!({ "model": "x", "messages": [{ "role": "user", "content": "hi" }] });
    let r = client.post(&url).json(&body).send().await.unwrap();
    assert_eq!(r.status(), 401, "POST 入口无 token 必须 401");
    let r = client
        .post(&url)
        .bearer_auth("wrong")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401, "POST 入口错 token 必须 401");
    // 正确 token 应通过鉴权（后续路由解析失败返回非 401，证明已过 check_auth）
    let r = client
        .post(&url)
        .bearer_auth("secret-token")
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_ne!(r.status(), 401, "正确 token 应通过鉴权");
}

/// 回归：插件短路直答不得绕过审计——usage 写入数据库 + trace 写入磁盘都要有。
/// 历史缺陷是这些路径直接 return，插件代答的请求在用量与 Traces 页完全不可见。
#[tokio::test]
async fn e2e_short_circuit_still_records_usage_and_trace() {
    let (state, db, trace_dir) = setup_state_lua(
        "sc",
        r#"
        MB = { version = "0.1.0", capabilities = { "raw_request" } }
        function MB.on_client_request_raw(ctx, msg)
          return {
            action = "short_circuit",
            status = 200,
            headers = { { "content-type", "application/json" } },
            body = { ok = true, cached_by = "e2e" },
          }
        end
        "#,
        &["raw_request"],
        // 若短路失效走到上游，客户端会看到 SHOULD-NOT-BE-REACHED
        json!({
            "id": "msg_x", "model": "claude-x", "role": "assistant",
            "content": [{ "type": "text", "text": "SHOULD-NOT-BE-REACHED" }],
            "stop_reason": "end_turn", "usage": { "input_tokens": 1, "output_tokens": 1 }
        }),
    )
    .await;

    let body = json!({
        "model": "test-model",
        "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let out: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(out["cached_by"], "e2e", "应返回插件给的报文: {out}");
    assert!(
        !out.to_string().contains("SHOULD-NOT-BE-REACHED"),
        "短路后不得再调上游"
    );

    let rows = db
        .query_usage(&UsageQuery {
            limit: 5,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rows.len(), 1, "短路直答也必须留下一条用量记录");
    assert_eq!(rows[0].status.as_deref(), Some("ok"), "2xx 直答记为 ok");

    let files = trace_json_files(&trace_dir);
    assert_eq!(files.len(), 1, "短路直答也必须留下一份 trace");
    let t: Value = serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
    assert_eq!(t["status"], "short_circuit", "trace 要能看出是谁答的: {t}");
    assert_eq!(t["clientResponse"]["cached_by"], "e2e");
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 回归：插件主动中止（abort）同样要留下用量与 trace，状态记 error。
#[tokio::test]
async fn e2e_abort_still_records_usage_and_trace() {
    let (state, db, trace_dir) = setup_state_lua(
        "abort",
        r#"
        MB = { version = "0.1.0", capabilities = { "raw_request" } }
        function MB.on_client_request_raw(ctx, msg)
          return { action = "abort", message = "e2e 主动中止" }
        end
        "#,
        &["raw_request"],
        json!({ "id": "msg_x", "content": [], "usage": {} }),
    )
    .await;

    let body = json!({
        "model": "test-model",
        "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let err = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect_err("abort 应让请求失败");
    let resp = err.into_response();
    assert!(!resp.status().is_success(), "中止必须是非 2xx");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(
        String::from_utf8_lossy(&bytes).contains("e2e 主动中止"),
        "插件给的中止消息应透传给客户端"
    );

    let rows = db
        .query_usage(&UsageQuery {
            limit: 5,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rows.len(), 1, "被中止的请求也要留下用量记录");
    assert_eq!(rows[0].status.as_deref(), Some("error"));

    let files = trace_json_files(&trace_dir);
    assert_eq!(files.len(), 1, "被中止的请求也要留下 trace");
    let t: Value = serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
    assert_eq!(t["status"], "aborted");
    assert_eq!(t["error"], "e2e 主动中止");
    let _ = std::fs::remove_dir_all(&trace_dir);
}

// ── 插件驱动的上游重试（5xx/520 自愈）────────────────────────────────────

/// 内联救援插件：上游错误路径（msg.error 且 >=500）一律要求宿主 1ms 后重试；
/// 成功路径（msg.error=false）返回 nil 放行。与 `plugins/utils/rescue_5xx.lua`
/// 同构（后者多了退避与计数封顶）；`msg.error` 由网关在非 2xx 时置位。
const RESCUE_PLUGIN: &str = r#"
MB = { version = "0.1.0", capabilities = { "raw_response" } }
function MB.on_upstream_response_raw(ctx, msg)
  if msg.error and msg.status >= 500 then
    return { action = "retry", delay_ms = 1 }
  end
end
"#;

/// 计数型 mock OpenAI Chat 上游：前 `fail_first` 次请求回 520（HTML 空壳，模拟
/// api.commandcode.ai 的瞬时 5xx），之后按请求 `stream` 返回正常 JSON / SSE。
///
/// 返回 (base_url, 命中计数)——命中次数即「宿主总共发了几次上游请求」，是断言
/// 「重试真的重发了」与「没重试」的唯一可信观察点（客户端只看得到末次结果）。
async fn spawn_mock_chat_520_then_ok(fail_first: usize) -> (String, Arc<AtomicUsize>) {
    type MockState = (Arc<AtomicUsize>, usize);
    async fn chat(
        axum::extract::State(s): axum::extract::State<MockState>,
        Json(body): Json<Value>,
    ) -> Response {
        let (hits, fail_first) = s;
        if hits.fetch_add(1, Ordering::SeqCst) < fail_first {
            return (
                axum::http::StatusCode::from_u16(520).unwrap(),
                "<html>520 服务端未知错误</html>",
            )
                .into_response();
        }
        if body
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            let sse = concat!(
                "data: {\"id\":\"c1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
                "data: {\"id\":\"c1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":4,\"total_tokens\":12}}\n\n",
                "data: [DONE]\n\n",
            );
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from(sse))
                .unwrap()
        } else {
            Json(json!({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [{ "index": 0, "message": { "role": "assistant", "content": "Hello from chat" }, "finish_reason": "stop" }],
                "usage": { "prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12 }
            }))
            .into_response()
        }
    }
    let hits = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route("/v1/chat/completions", post(chat))
        .with_state((hits.clone(), fail_first));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), hits)
}

/// 重试测试装配：计数型 520 上游（openai-chat）+ 内联救援插件 + 可调 `plugin_retry_max`。
///
/// 不复用 `setup_with`/`seed_lua_state`：前者无插件与 trace，后者把上游协议钉死在
/// anthropic 且用默认重试配置——本组测试三项都要能改。
async fn setup_retry_state(
    tag: &str,
    plugin_script: &str,
    retry_max: usize,
    fail_first: usize,
) -> (
    Arc<AppState>,
    Arc<Database>,
    std::path::PathBuf,
    Arc<AtomicUsize>,
) {
    let cfg = GatewayConfig {
        plugin_retry_max: retry_max,
        ..GatewayConfig::default()
    };
    setup_retry_state_with(tag, plugin_script, fail_first, cfg).await
}

/// `setup_retry_state` 的可定制版：调用方给 `GatewayConfig`（`trace_dir` 由本函数
/// 按 tag 分配并回填），其余装配（mock 上游、provider/route/插件种子）不变。
async fn setup_retry_state_with(
    tag: &str,
    plugin_script: &str,
    fail_first: usize,
    mut cfg: GatewayConfig,
) -> (
    Arc<AppState>,
    Arc<Database>,
    std::path::PathBuf,
    Arc<AtomicUsize>,
) {
    let (base_url, hits) = spawn_mock_chat_520_then_ok(fail_first).await;
    let db = Arc::new(Database::open_in_memory().unwrap());
    db.upsert_provider(&Provider {
        key: "mock".into(),
        endpoints: vec![Endpoint {
            protocol: "openai-chat".into(),
            base_url,
            api_key: "sk-test".into(),
        }],
        version: None,
        user_agent: None,
        web_search: None,
        extra: json!({}),
        quota_plugin_ref: String::new(),
        quota_interval_secs: 0,
        quota_enabled: false,
        quota_config: serde_json::json!({}),
        enabled: true,
        created_at: 0,
        updated_at: 0,
    })
    .unwrap();
    db.upsert_route(&Route {
        alias: "test-model".into(),
        model_slug: "gpt-4o".into(),
        provider_key: "mock".into(),
        extra: Value::Null,
    })
    .unwrap();
    db.upsert_plugin(&PluginRecord {
        name: format!("e2e-{tag}"),
        source: "lua".into(),
        script_ref: plugin_script.into(),
        enabled: true,
        config: json!({}),
        scopes: vec!["global".into()],
        capabilities: vec!["raw_response".into()],
        category: "core".into(),
        config_schema: Value::Null,
    })
    .unwrap();

    let trace_dir = std::env::temp_dir().join(format!("mb-e2e-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&trace_dir);
    cfg.trace_dir = Some(trace_dir.to_string_lossy().to_string());
    let state = bootstrap(cfg, db.clone()).expect("bootstrap 应成功加载插件");
    (state, db, trace_dir, hits)
}

/// 520 自愈（非流式）：前 2 次 520、第 3 次 200 ⇒ 客户端拿到 200，上游命中 3 次。
///
/// 这是本次改造的目标场景：单端点 provider 的瞬时 5xx 不再冒泡到客户端。
#[tokio::test]
async fn e2e_plugin_retry_recovers_non_stream_520() {
    let (state, db, trace_dir, hits) = setup_retry_state("retry-ok", RESCUE_PLUGIN, 8, 2).await;

    let body = json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let resp = dispatch::handle_request(state, Protocol::OpenAiChat, body, vec![], None)
        .await
        .expect("插件重试后应成功");
    assert_eq!(resp.status(), 200);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let out: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        out["object"], "chat.completion",
        "客户端拿到正常响应: {out}"
    );
    assert_eq!(
        untag(out["choices"][0]["message"]["content"].as_str().unwrap()),
        "Hello from chat"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        3,
        "两次 520 后第三次成功 ⇒ 上游应被命中 3 次"
    );
    assert_eq!(
        db.usage_summary().unwrap().requests,
        1,
        "一次客户端请求只记一条用量（重试轮不单独记账）"
    );

    let files = trace_json_files(&trace_dir);
    assert_eq!(files.len(), 1, "重试轮不落 trace，只留末次");
    let t: Value = serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
    assert_eq!(t["status"], "ok");
    assert_eq!(t["retries"], 2, "两次重试要写进 trace: {t}");
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 520 自愈（流式）：同样的 2 次 520 + 第 3 次成功，SSE 照常完整下发。
///
/// 流式请求的错误响应在响应头阶段就被拦住（尚未下发任何 SSE 事件），所以重试对
/// 客户端完全透明：它只会看到第三次尝试的 SSE 流。
#[tokio::test]
async fn e2e_plugin_retry_recovers_stream_520() {
    let (state, db, trace_dir, hits) = setup_retry_state("retry-sse", RESCUE_PLUGIN, 8, 2).await;

    let body = json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::OpenAiChat, body, vec![], None)
        .await
        .expect("流式请求同样应自愈");
    assert_eq!(resp.status(), 200);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let sse = String::from_utf8_lossy(&bytes);
    assert!(sse.contains("Hello"), "应下发正常增量: {sse}");
    assert!(sse.contains("[DONE]"), "SSE 应以 [DONE] 收尾: {sse}");
    assert_eq!(hits.load(Ordering::SeqCst), 3, "流式同样重发 3 次");
    assert_eq!(db.usage_summary().unwrap().requests, 1);

    // 流式 trace 在 StreamAudit 析构时落盘（body 读尽后），留一点时间等它写完
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let files = trace_json_files(&trace_dir);
    assert_eq!(files.len(), 1, "流式请求应留下一份 trace");
    let t: Value = serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
    assert_eq!(t["retries"], 2, "流式路径的重试次数也要落盘: {t}");
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 永远 520：重试预算耗尽后如实报错——客户端最终拿到 520，上游命中 = 1 + 重试次数。
///
/// 上限由宿主强制（插件每轮都返回 retry），这条同时守着「不会死循环」。
#[tokio::test]
async fn e2e_plugin_retry_exhaustion_reports_upstream_520() {
    let (state, db, trace_dir, hits) =
        setup_retry_state("retry-give-up", RESCUE_PLUGIN, 2, usize::MAX).await;

    let body = json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let err = dispatch::handle_request(state, Protocol::OpenAiChat, body, vec![], None)
        .await
        .expect_err("重试耗尽后应把上游错误如实报出");
    assert!(
        matches!(
            &err,
            moonbridge_gateway::GatewayError::Upstream { status: 520, .. }
        ),
        "客户端应拿到上游的 520: {err}"
    );
    assert_eq!(
        err.into_response().status(),
        520,
        "回给客户端的 HTTP 状态码也必须是 520"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        3,
        "plugin_retry_max=2 ⇒ 1 次首试 + 2 次重试"
    );
    assert_eq!(db.usage_summary().unwrap().requests, 1);

    let files = trace_json_files(&trace_dir);
    assert_eq!(files.len(), 1);
    let t: Value = serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
    assert_eq!(t["status"], "error");
    assert_eq!(t["retries"], 2, "失败请求的重试次数同样要留痕: {t}");
    // trace.error 的文案沿用既有格式；520 不在 http crate 的已知码表里，Display 会补
    // 「<unknown status code>」后缀（历史行为，未改），故只断言前缀。
    assert!(
        t["error"]
            .as_str()
            .unwrap_or_default()
            .starts_with("上游返回 HTTP 520"),
        "错误摘要应含上游状态码: {t}"
    );
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 流内零产出失败自愈（cctq/new-api 实测形态）：上游 200 + `error` 事件帧宣告失败，
/// 客户端还什么都没收到——插件错误钩子返回 retry，网关重发，客户端最终拿到正常流。
#[tokio::test]
async fn e2e_plugin_retry_recovers_stream_inband_failure() {
    /// 命中 `msg.error` 就重试——流内失败的合成报文 status=200、error=true。
    const RETRY_ON_ANY_ERROR_PLUGIN: &str = r#"
MB = { version = "0.1.0", capabilities = { "raw_response" } }
function MB.on_upstream_response_raw(ctx, msg)
  if msg.error then return { action = "retry", delay_ms = 1 } end
end
"#;
    // 计数型 mock：首击回 200 + error 帧（上游宣告失败），第二击回正常流。
    async fn handler(
        axum::extract::State((hits, _)): axum::extract::State<(Arc<AtomicUsize>, ())>,
    ) -> Response {
        let sse = if hits.fetch_add(1, Ordering::SeqCst) == 0 {
            concat!(
                "event: message_start\n",
                "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_x\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":5,\"output_tokens\":0}}}\n\n",
                "event: error\n",
                "data: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"Upstream request failed\"}}\n\n",
            )
        } else {
            concat!(
                "event: message_start\n",
                "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
                "event: content_block_start\n",
                "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
                "event: content_block_stop\n",
                "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "event: message_delta\n",
                "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":4}}\n\n",
                "event: message_stop\n",
                "data: {\"type\":\"message_stop\"}\n\n",
            )
        };
        Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from(sse))
            .unwrap()
    }
    let hits = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route("/v1/messages", post(handler))
        .with_state((hits.clone(), ()));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let base_url = format!("http://{addr}");

    let (state, db, trace_dir) = seed_lua_state(
        "retry-inband",
        RETRY_ON_ANY_ERROR_PLUGIN,
        &["raw_response"],
        base_url,
    )
    .await;
    let text = collect_stream(state).await;

    assert_eq!(hits.load(Ordering::SeqCst), 2, "零产出失败应重发一次");
    assert!(
        text.contains("Hello"),
        "客户端应拿到重试后的正常内容: {text}"
    );
    assert!(
        text.contains("message_stop"),
        "重试后的流应正常收尾: {text}"
    );
    assert_eq!(db.usage_summary().unwrap().requests, 1, "重试轮不单独记账");

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let files = trace_json_files(&trace_dir);
    assert_eq!(files.len(), 1);
    let t: Value = serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
    assert_eq!(t["status"], "ok", "{t}");
    assert_eq!(t["retries"], 1, "流内失败的重试次数要留痕: {t}");
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 流内零产出失败 + 插件放行：错误帧照常透传客户端并记 error（不吞）。
#[tokio::test]
async fn e2e_stream_inband_failure_passes_through_when_plugin_passes() {
    /// 只观测、永不重试。
    const OBSERVE_PLUGIN: &str = r#"
MB = { version = "0.1.0", capabilities = { "raw_response" } }
function MB.on_upstream_response_raw(ctx, msg)
  return nil
end
"#;
    let sse = concat!(
        "event: error\n",
        "data: {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"Upstream request failed\"}}\n\n",
    );
    let (state, db, _trace_dir) = seed_lua_state(
        "inband-pass",
        OBSERVE_PLUGIN,
        &["raw_response"],
        spawn_mock_sse_body(sse).await,
    )
    .await;
    let text = collect_stream(state).await;
    assert!(
        text.contains("\"type\":\"error\""),
        "插件放行时错误帧应透传客户端: {text}"
    );
    let rows = db
        .query_usage(&UsageQuery {
            limit: 5,
            ..Default::default()
        })
        .unwrap();
    assert_eq!(rows[0].status.as_deref(), Some("error"));
}

/// 非流式 `request_timeout_secs` 是整条请求的总预算，插件重试必须共享它：
/// 剩余预算装不下一次重试延迟时放弃重试、把真实上游错误还给客户端。
///
/// 回归：旧实现每次发送都拿全额 `request_timeout_secs`，重试+睡眠累计后整条请求
/// 超出总预算仍继续重发（本例旧实现会发出第 3 次请求）；响应体未及时就绪时，
/// 上游的 200 还会被 `non_stream` 的读体剩余预算（已耗尽）改判成 504。
/// 本例 fail_first=2：首试 520 → 预算内允许一次 600ms 重试 → 第二次 520 时
/// 剩余预算 < 600ms，放弃重试 ⇒ 客户端 520、上游命中 2 次。
#[tokio::test]
async fn e2e_plugin_retry_respects_request_timeout_budget() {
    const SLOW_RESCUE_PLUGIN: &str = r#"
MB = { version = "0.1.0", capabilities = { "raw_response" } }
function MB.on_upstream_response_raw(ctx, msg)
  if msg.error and msg.status >= 500 then
    return { action = "retry", delay_ms = 600 }
  end
end
"#;
    let cfg = GatewayConfig {
        request_timeout_secs: 1,
        ..GatewayConfig::default()
    };
    let (state, db, trace_dir, hits) =
        setup_retry_state_with("retry-budget", SLOW_RESCUE_PLUGIN, 2, cfg).await;

    let body = json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let err = dispatch::handle_request(state, Protocol::OpenAiChat, body, vec![], None)
        .await
        .expect_err("剩余预算不足应放弃重试、放行原始上游错误");
    assert!(
        matches!(
            &err,
            moonbridge_gateway::GatewayError::Upstream { status: 520, .. }
        ),
        "应回传上游真实的 520 而非人造 504: {err}"
    );
    assert_eq!(
        err.into_response().status(),
        520,
        "回给客户端的 HTTP 状态码也必须是 520"
    );
    assert_eq!(
        hits.load(Ordering::SeqCst),
        2,
        "首试 + 1 次预算内重试后即放弃，上游应被命中 2 次"
    );
    assert_eq!(db.usage_summary().unwrap().requests, 1);
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// `plugin_retry_max = 0`：插件重试整体禁用，行为与改造前完全一致（一次都不重发）。
#[tokio::test]
async fn e2e_plugin_retry_disabled_is_legacy_behaviour() {
    let (state, db, trace_dir, hits) =
        setup_retry_state("retry-off", RESCUE_PLUGIN, 0, usize::MAX).await;

    let body = json!({
        "model": "test-model",
        "messages": [{ "role": "user", "content": "Hi" }],
        "stream": false
    });
    let err = dispatch::handle_request(state, Protocol::OpenAiChat, body, vec![], None)
        .await
        .expect_err("禁用重试后 520 直接冒泡");
    assert!(matches!(
        &err,
        moonbridge_gateway::GatewayError::Upstream { status: 520, .. }
    ));
    assert_eq!(
        hits.load(Ordering::SeqCst),
        1,
        "plugin_retry_max=0 ⇒ 禁用插件重试，上游只被命中一次"
    );
    assert_eq!(db.usage_summary().unwrap().requests, 1);

    let files = trace_json_files(&trace_dir);
    assert_eq!(files.len(), 1);
    let t: Value = serde_json::from_slice(&std::fs::read(&files[0]).unwrap()).unwrap();
    assert!(
        t.get("retries").is_none(),
        "0 次重试不写 retries 键（旧 trace 形态）: {t}"
    );
    let _ = std::fs::remove_dir_all(&trace_dir);
}

// ── 会话水印（session marker）──────────────────────────────────────────

/// 水印测试装配：固定 JSON mock 上游 + trace 写入磁盘 + 可切换 `session_marker`/表深。
///
/// 不复用 `setup_with`：它固定 `GatewayConfig::default()`。这里需要 trace——它是唯一
/// 能同时看到「客户端送来的报文」(`clientRequest`) 与「转发上游的报文」
/// (`upstreamRequest.body`) 的观察点，而「剥除干净后才转发」正是本特性的核心不变量。
async fn setup_state_marker(
    tag: &str,
    session_marker: bool,
    session_table_depth: usize,
    mock_body: Value,
) -> (Arc<AppState>, Arc<Database>, std::path::PathBuf) {
    let base_url = spawn_mock_returning(mock_body).await;
    let db = Arc::new(Database::open_in_memory().unwrap());
    db.upsert_provider(&Provider {
        key: "mock".into(),
        endpoints: vec![Endpoint {
            protocol: "anthropic".into(),
            base_url,
            api_key: "sk-test".into(),
        }],
        version: Some("2023-06-01".into()),
        user_agent: None,
        web_search: None,
        extra: json!({}),
        quota_plugin_ref: String::new(),
        quota_interval_secs: 0,
        quota_enabled: false,
        quota_config: serde_json::json!({}),
        enabled: true,
        created_at: 0,
        updated_at: 0,
    })
    .unwrap();
    db.upsert_route(&Route {
        alias: "test-model".into(),
        model_slug: "claude-x".into(),
        provider_key: "mock".into(),
        extra: Value::Null,
    })
    .unwrap();
    let trace_dir = std::env::temp_dir().join(format!("mb-e2e-sess-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&trace_dir);
    let cfg = GatewayConfig {
        trace_dir: Some(trace_dir.to_string_lossy().to_string()),
        session_marker,
        session_table_depth,
        ..GatewayConfig::default()
    };
    let state = bootstrap(cfg, db.clone()).expect("bootstrap 应成功");
    (state, db, trace_dir)
}

/// 带 thinking 块的 Anthropic 非流式响应——CoT 水印的载体（无推理块则不打标记）。
fn thinking_mock_body() -> Value {
    json!({
        "id": "msg_1", "model": "claude-x", "role": "assistant",
        "content": [
            { "type": "thinking", "thinking": "hmm", "signature": "sig_mock" },
            { "type": "text", "text": "Hello from mock" }
        ],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 10, "output_tokens": 5 }
    })
}

/// 发一次非流式 Anthropic 请求，返回客户端收到的完整响应报文。
async fn post_anthropic(state: Arc<AppState>, body: Value, session_id: Option<String>) -> Value {
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], session_id)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// 取 thinking 明文首部的水印载荷（`" [mb:<载荷>]…"`）；无水印或形态不合法返回 `None`。
fn tag_of(thinking: &str) -> Option<String> {
    let inner = thinking.strip_prefix(' ')?.strip_prefix("[mb:")?;
    let end = inner.find(']')?;
    let p = &inner[..end];
    (p.len() == 36 || p.len() == 6).then(|| p.to_string())
}

/// 从 SSE 文本里取首个 `[mb:<载荷>]`（marker 以推理明文增量下发、位于 thinking 首部）。
fn sse_tag(sse: &str) -> Option<String> {
    let i = sse.find("[mb:")?;
    let rest = &sse[i + 4..];
    let end = rest.find(']')?;
    let p = &rest[..end];
    (p.len() == 36 || p.len() == 6).then(|| p.to_string())
}

/// 读取目录下全部 trace 文件（顺序不保证，断言须与顺序无关）。
fn traces(dir: &std::path::Path) -> Vec<Value> {
    trace_json_files(dir)
        .iter()
        .map(|p| serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap())
        .collect()
}

/// thinking + signature_delta + text 的 Anthropic SSE 序列（CoT 水印的流式载体）。
fn sse_thinking_then_text() -> String {
    concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":4,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_mock\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    )
    .to_string()
}

/// thinking + signature_delta + tool_use 的 Anthropic SSE 序列（工具轮的 CoT 水印）。
fn sse_thinking_then_tool_use() -> String {
    concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"model\":\"claude-x\",\"usage\":{\"input_tokens\":4,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_mock\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tu_1\",\"name\":\"search\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"q\\\":\\\"x\\\"}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":3}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    )
    .to_string()
}

/// 逐帧解析 Anthropic SSE（不依赖 serde 键序）。
fn anthropic_frames(sse: &str) -> Vec<Value> {
    sse.lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .filter_map(|d| serde_json::from_str::<Value>(d.trim()).ok())
        .collect()
}

/// 取 index 块的全部 thinking_delta 明文（按出现顺序）。
fn thinking_deltas(frames: &[Value], index: u64) -> Vec<String> {
    frames
        .iter()
        .filter(|f| {
            f["type"] == "content_block_delta"
                && f["index"].as_u64() == Some(index)
                && f["delta"]["type"] == "thinking_delta"
        })
        .map(|f| f["delta"]["thinking"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn e2e_session_marker_round_trips_and_never_reaches_upstream() {
    let (state, _db, trace_dir) =
        setup_state_marker("roundtrip", true, 64, thinking_mock_body()).await;

    // 第 1 轮：请求不带 marker ⇒ 新分配会话；水印嵌进 thinking 明文首部，
    // 用户可见正文不携带任何 marker。
    let first = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }], "stream": false
    });
    let out1 = post_anthropic(state.clone(), first, None).await;
    assert_eq!(out1["content"][0]["type"], "thinking");
    let think1 = out1["content"][0]["thinking"].as_str().unwrap().to_string();
    let tag1 = tag_of(&think1).expect("首轮 thinking 响应应带水印");
    assert_eq!(tag1.len(), 36, "自分配会话的 marker 载荷是 uuid: {tag1}");
    assert!(think1.ends_with("hmm"), "水印不得损伤推理原文: {think1}");
    assert_eq!(out1["content"][1]["text"], "Hello from mock", "正文块完好");
    assert!(
        !out1["content"][1].to_string().contains("mb:"),
        "用户可见正文不得出现 marker: {}",
        out1["content"][1]
    );

    // 第 2 轮：客户端把整段历史原样带回（thinking 块含水印 + signature）⇒ 同一会话
    let second = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [
            { "role": "user", "content": "Hi" },
            { "role": "assistant", "content": out1["content"].clone() },
            { "role": "user", "content": "Again" }
        ],
        "stream": false
    });
    let out2 = post_anthropic(state.clone(), second, None).await;
    let think2 = out2["content"][0]["thinking"].as_str().unwrap();
    assert_eq!(
        tag_of(think2).as_deref(),
        Some(tag1.as_str()),
        "同一会话应复用同一 marker 载荷: {think2}"
    );

    // 第 3 轮：带回一个合法但不在表里的短 tag（陈旧外部 tag / 跨版本历史）⇒
    // 载荷就地成确定性身份 `mb-{tag}`，且 marker 原样回发、形态恒定
    let third = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [
            { "role": "user", "content": "Hi" },
            { "role": "assistant", "content": [
                { "type": "thinking", "thinking": " [mb:abcdef]stale", "signature": "sig_old" },
                { "type": "text", "text": "old" }
            ] },
            { "role": "user", "content": "Fresh" }
        ],
        "stream": false
    });
    let out3 = post_anthropic(state, third, None).await;
    let think3 = out3["content"][0]["thinking"].as_str().unwrap();
    assert_eq!(
        tag_of(think3).as_deref(),
        Some("abcdef"),
        "陈旧短 tag 应原样回发、marker 形态恒定: {think3}"
    );

    let tr = traces(&trace_dir);
    assert_eq!(tr.len(), 3, "三问各留一份 trace");
    // 按各自的水印定位 trace，不依赖写入顺序
    let sid_of = |pat: &str| {
        tr.iter()
            .find(|t| t["clientRequest"].to_string().contains(pat))
            .unwrap_or_else(|| panic!("应有含 {pat} 的入站 trace"))
            .pointer("/sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string()
    };
    let sid1 = tr
        .iter()
        .find(|t| !t["clientRequest"].to_string().contains("[mb:"))
        .and_then(|t| t["sessionId"].as_str())
        .unwrap_or("")
        .to_string();
    let sid2 = sid_of(&format!("[mb:{tag1}]"));
    let sid3 = sid_of("[mb:abcdef]");
    assert!(!sid1.is_empty(), "水印生效时 session 不得为空");
    assert_eq!(sid1, sid2, "前两问应同会话（marker 载荷即身份）");
    assert_eq!(sid2, tag1, "uuid 载荷的 session id 就是载荷本身");
    assert_eq!(sid3, "mb-abcdef", "陈旧短 tag 合成确定性身份");

    // 核心不变量：marker 只活在客户端存档里，转发上游的报文必须干净
    for t in &tr {
        let up = t["upstreamRequest"]["body"].to_string();
        assert!(!up.contains("mb:"), "转发上游的报文不得含水印: {up}");
    }
    // 带着 marker 进来的那两问，trace 要如实记录客户端原文
    let with_marker = tr
        .iter()
        .filter(|t| t["clientRequest"].to_string().contains("[mb:"))
        .count();
    assert_eq!(
        with_marker, 2,
        "第 2、3 问的入站报文应照实含 marker（trace 记的是客户端原样）"
    );

    // 逐问确认「推理原文还原、水印剥除干净」
    let t2 = tr
        .iter()
        .find(|t| {
            t["clientRequest"]
                .to_string()
                .contains(&format!("[mb:{tag1}]"))
        })
        .expect("第 2 问的入站报文应带首轮水印");
    let up2 = t2["upstreamRequest"]["body"].to_string();
    assert!(
        up2.contains("hmm"),
        "剥水印后 thinking 原文应还原上行: {up2}"
    );
    assert!(
        up2.contains("Hello from mock"),
        "剥水印不得连带剥掉历史正文: {up2}"
    );
    assert!(!up2.contains("mb:"), "第 2 问转发上游须干净: {up2}");

    let t3 = tr
        .iter()
        .find(|t| t["clientRequest"].to_string().contains("[mb:abcdef]"))
        .expect("第 3 问带回的陈旧水印也要被认出并剥除");
    let up3 = t3["upstreamRequest"]["body"].to_string();
    assert!(up3.contains("stale"), "陈旧 marker 的宿主推理应保留: {up3}");
    assert!(!up3.contains("mb:"), "第 3 问转发上游须干净: {up3}");
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 插件会话覆写：`on_client_request_raw` 写 `msg.session_id` 是最高优先级
/// 身份源——客户端自带会话头（`x-opencode-session`）经插件转为 session，
/// 水印 marker 降级为不带头时的回退方案（生产问题：剥离 thinking 的客户端
/// 让 marker 断链、亲和头逐请求漂移）。
#[tokio::test]
async fn e2e_plugin_session_id_overrides_watermark() {
    let (state, _db, trace_dir) = setup_state_lua(
        "sess-override",
        include_str!("../../../plugins/harness/opencode.lua"),
        &["raw_request"],
        thinking_mock_body(),
    )
    .await;

    // 带头请求：客户端身份胜出——不带 marker 也归入它的会话
    let body = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }], "stream": false
    });
    let headers = vec![("x-opencode-session".to_string(), "oc-sess-123".to_string())];
    let resp = dispatch::handle_request(state.clone(), Protocol::Anthropic, body, headers, None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let out: Value = serde_json::from_slice(&bytes).unwrap();

    let tr = traces(&trace_dir);
    assert_eq!(tr.len(), 1);
    assert_eq!(
        tr[0]["sessionId"], "oc-sess-123",
        "插件覆写的 session_id 应成为 ctx.session_id"
    );
    // 外部身份的 marker 载荷是 6-hex 短 tag（tag_from_id），非 uuid
    let think = out["content"][0]["thinking"].as_str().unwrap();
    let tag = tag_of(think).expect("响应应打水印");
    assert_eq!(tag.len(), 6, "外部身份的载荷应是短 tag: {tag}");

    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 活跃表淘汰/进程重启不再改判会话身份：uuid 载荷自带身份，带回即还原——
/// 这正是长对话中 `x-opencode-session` 等下游亲和头不再漂移的保证。
#[tokio::test]
async fn e2e_session_marker_identity_survives_table_eviction() {
    let (state, _db, trace_dir) = setup_state_marker("evict", true, 1, thinking_mock_body()).await;

    let mk = |who: &str| {
        json!({
            "model": "test-model", "max_tokens": 64,
            "messages": [{ "role": "user", "content": who }], "stream": false
        })
    };
    // 第 1 轮：新会话 u1；第 2 轮：另一新会话 u2 —— 表深 1 把 u1 淘汰
    let out1 = post_anthropic(state.clone(), mk("a"), None).await;
    let u1 = tag_of(out1["content"][0]["thinking"].as_str().unwrap()).expect("首轮应打标记");
    let out2 = post_anthropic(state.clone(), mk("b"), None).await;
    let u2 = tag_of(out2["content"][0]["thinking"].as_str().unwrap()).expect("第二轮应打标记");
    assert_ne!(u1, u2, "两次新分配应是不同会话");

    // 第 3 轮：带回 u1 的 marker（u1 已被淘汰出表）⇒ 仍落 u1，而非另派新会话
    let third = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [
            { "role": "user", "content": "a" },
            { "role": "assistant", "content": out1["content"].clone() },
            { "role": "user", "content": "back" }
        ],
        "stream": false
    });
    let out3 = post_anthropic(state, third, None).await;
    assert_eq!(
        tag_of(out3["content"][0]["thinking"].as_str().unwrap()).as_deref(),
        Some(u1.as_str()),
        "淘汰后带回同一 marker 仍须复用同一载荷"
    );

    let tr = traces(&trace_dir);
    let t3 = tr
        .iter()
        .find(|t| {
            t["clientRequest"]
                .to_string()
                .contains(&format!("[mb:{u1}]"))
        })
        .expect("第 3 问的入站报文应带 u1 水印");
    assert_eq!(
        t3["sessionId"].as_str().unwrap(),
        u1,
        "淘汰后带回同一 marker 仍须同一会话身份"
    );
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 新方案核心：tool_use 轮也能打标记——marker 嵌推理块首部，与工具调用共存。
#[tokio::test]
async fn e2e_session_marker_tags_tool_call_round_via_reasoning() {
    let (state, _db, trace_dir) = setup_state_marker(
        "toolcall",
        true,
        64,
        json!({
            "id": "msg_t", "model": "claude-x", "role": "assistant",
            "content": [
                { "type": "thinking", "thinking": "hmm", "signature": "sig_t" },
                { "type": "text", "text": "sure" },
                { "type": "tool_use", "id": "tu_1", "name": "search", "input": { "q": "x" } }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 7, "output_tokens": 3 }
        }),
    )
    .await;

    let body = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }], "stream": false
    });
    let out = post_anthropic(state, body, None).await;
    let tag = tag_of(out["content"][0]["thinking"].as_str().unwrap())
        .expect("tool_use 轮的 thinking 也应打标记");
    assert_eq!(tag.len(), 36);
    assert_eq!(out["content"][1]["text"], "sure", "正文块完好");
    assert_eq!(out["content"][2]["type"], "tool_use", "工具块完好");
    assert!(
        !out["content"][1].to_string().contains("mb:")
            && !out["content"][2].to_string().contains("mb:"),
        "正文与工具块不得出现 marker"
    );
    let tr = traces(&trace_dir);
    assert_eq!(tr.len(), 1);
    assert_eq!(
        tr[0]["sessionId"].as_str().unwrap(),
        tag,
        "会话 id 即 marker 载荷"
    );
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 纯 CoT 方案对照：无推理块的轮次不打标记——不向正文插入无关文本、不凭空造块；
/// 但会话照常解析（session id 仍在），身份顺延到下一个含推理块的响应。
#[tokio::test]
async fn e2e_session_marker_absent_without_reasoning() {
    let (state, _db, trace_dir) = setup_state_marker(
        "noreason",
        true,
        64,
        json!({
            "id": "msg_t", "model": "claude-x", "role": "assistant",
            "content": [
                { "type": "text", "text": "sure" },
                { "type": "tool_use", "id": "tu_1", "name": "search", "input": { "q": "x" } }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 7, "output_tokens": 3 }
        }),
    )
    .await;

    let body = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }], "stream": false
    });
    let out = post_anthropic(state, body, None).await;
    let raw = out.to_string();
    assert!(!raw.contains("mb:"), "无推理块的响应不得打标记: {raw}");
    assert_eq!(out["content"][0]["text"], "sure", "正文块应完好");
    assert_eq!(out["content"][1]["type"], "tool_use", "工具块应完好");
    let tr = traces(&trace_dir);
    assert_eq!(tr.len(), 1);
    assert_ne!(tr[0]["sessionId"], Value::Null, "无水印轮也要有 session id");
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 关掉水印：不再嵌入，但**入站 marker 照剥**——客户端可能带着开启期间留下的历史。
#[tokio::test]
async fn e2e_session_marker_off_still_strips_inbound() {
    let (state, _db, trace_dir) = setup_state_marker(
        "off",
        false,
        64,
        json!({
            "id": "msg_1", "model": "claude-x", "role": "assistant",
            "content": [
                { "type": "thinking", "thinking": "hmm", "signature": "sig_mock" },
                { "type": "text", "text": "Hello from mock" }
            ],
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 10, "output_tokens": 5 }
        }),
    )
    .await;

    let body = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [
            { "role": "user", "content": "Hi" },
            // 旧式正文尾部 marker 与新版 CoT 首部 marker 都要剥
            { "role": "assistant", "content": [
                { "type": "thinking", "thinking": " [mb:998877]old think", "signature": "sig_old" },
                { "type": "text", "text": "Hello from mock [mb:12ab34]" }
            ] },
            { "role": "user", "content": "Again" }
        ],
        "stream": false
    });
    let out = post_anthropic(state, body, None).await;
    assert!(
        !out.to_string().contains("mb:"),
        "关闭水印时不得嵌入: {out}"
    );

    let tr = traces(&trace_dir);
    assert_eq!(tr.len(), 1);
    let up = tr[0]["upstreamRequest"]["body"].to_string();
    assert!(
        !up.contains("mb:"),
        "即使关闭水印，入站 marker 仍须剥除干净: {up}"
    );
    assert!(up.contains("Hello from mock"), "剥水印不得剥掉历史正文");
    assert!(up.contains("old think"), "剥水印不得剥掉推理原文");
    assert!(tr[0]["clientRequest"].to_string().contains("[mb:12ab34]"));
    assert_eq!(
        tr[0]["sessionId"],
        Value::Null,
        "关闭水印且无外部身份时应回到无会话（与改动前一致）"
    );
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 外部身份（body `session_id` / 头）优先于 marker：同一外部 id 必须稳定落同一会话。
#[tokio::test]
async fn e2e_external_session_id_wins_over_marker() {
    let (state, _db, trace_dir) =
        setup_state_marker("external", true, 64, thinking_mock_body()).await;

    let mk = |who: &str| {
        json!({
            "model": "test-model", "max_tokens": 64,
            "messages": [{ "role": "user", "content": who }], "stream": false
        })
    };
    // 外部 id 可以不是 uuid：marker 载荷是定长 6-hex 短 tag，经活跃表还原外部 id
    let out1 = post_anthropic(state.clone(), mk("a"), Some("ext-sess-1".into())).await;
    let think1 = out1["content"][0]["thinking"].as_str().unwrap().to_string();
    let tag1 = tag_of(&think1).expect("外部会话也要能回水印");
    assert_eq!(tag1.len(), 6, "外部身份的载荷应是短 tag: {tag1}");
    let out2 = post_anthropic(state, mk("b"), Some("ext-sess-1".into())).await;
    assert_eq!(
        tag_of(out2["content"][0]["thinking"].as_str().unwrap()).as_deref(),
        Some(tag1.as_str()),
        "同一外部 id 应稳定映射同一 tag"
    );

    let tr = traces(&trace_dir);
    assert_eq!(tr.len(), 2);
    let sids: Vec<&str> = tr
        .iter()
        .map(|t| t["sessionId"].as_str().unwrap_or(""))
        .collect();
    assert_eq!(sids[0], sids[1], "外部 id 未被 marker 改判");
    assert_eq!(sids[0], "ext-sess-1", "外部 id 应原样作为 session id");
    assert!(
        !tr.iter()
            .any(|t| t["upstreamRequest"]["body"].to_string().contains("mb:")),
        "转发上游仍须干净"
    );
    let _ = std::fs::remove_dir_all(&trace_dir);
}

/// 流式：水印嵌进首个推理块的**明文首部**——首个 thinking_delta 即 marker，
/// 不新占块、不动块序，signature_delta 照常随行。
#[tokio::test]
async fn e2e_session_marker_stream_embeds_in_thinking_head() {
    let base_url = spawn_mock_sse(sse_thinking_then_text()).await;
    let (state, _db) = setup_with("anthropic", base_url, "claude-x", "test-model").await;

    let body = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }], "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let sse = String::from_utf8_lossy(&bytes).to_string();

    assert!(sse.contains("Hello"), "正文增量不受影响: {sse}");
    let tag = sse_tag(&sse).expect("含 thinking 的流应带水印: {sse}");
    assert_eq!(tag.len(), 36);

    let frames = anthropic_frames(&sse);
    let td = thinking_deltas(&frames, 0);
    assert_eq!(
        td.first().map(String::as_str),
        Some(format!(" [mb:{tag}]").as_str()),
        "marker 应是 thinking 块的首部增量: {td:?}"
    );
    assert_eq!(
        td.get(1).map(String::as_str),
        Some("hmm"),
        "上游推理原文随行其后: {td:?}"
    );
    assert!(
        frames
            .iter()
            .any(|f| f["delta"]["type"] == "signature_delta"),
        "signature_delta 应照常随行: {sse}"
    );
    // 不新增独立 marker 块：块序号只有 0（thinking）与 1（text）
    let starts: Vec<u64> = frames
        .iter()
        .filter(|f| f["type"] == "content_block_start")
        .map(|f| f["index"].as_u64().unwrap())
        .collect();
    assert_eq!(starts, vec![0, 1], "marker 不得自成一块: {sse}");
    assert_eq!(
        frames.last().and_then(|f| f["type"].as_str()),
        Some("message_stop"),
        "末帧仍是 message_stop: {sse}"
    );
}

/// 流式跨协议回归：Chat 上游（reasoning_content 裸增量，无独立 BlockStart
/// 源自上游协议）× Anthropic 入口——且本轮含 tool_calls。marker 仍落在
/// thinking 首部：CoT 载体下工具轮照常打标记。
#[tokio::test]
async fn e2e_session_marker_stream_chat_upstream_reasoning_head() {
    let (base_url, _captured) = spawn_mock_chat_thinking_capture().await;
    let (state, _db) = setup_with("openai-chat", base_url, "deepseek-x", "test-model").await;

    let body = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }], "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let sse = String::from_utf8_lossy(&bytes).to_string();

    let tag = sse_tag(&sse).expect("chat 上游的 reasoning 流也应带水印: {sse}");
    let frames = anthropic_frames(&sse);
    let td = thinking_deltas(&frames, 0);
    assert_eq!(
        td.first().map(String::as_str),
        Some(format!(" [mb:{tag}]").as_str()),
        "marker 应是 thinking 首部增量: {td:?}"
    );
    assert!(
        td.iter().any(|t| t == "think hard"),
        "上游推理原文随行: {td:?}"
    );
    assert!(
        frames.iter().any(|f| f["type"] == "content_block_start"
            && f["content_block"]["type"] == "tool_use"),
        "tool_use 块应完整转发: {sse}"
    );
}

/// 流式跨协议回归：Responses 入口 × Chat thinking 上游。marker 随
/// `response.reasoning_summary_text.delta` 下发，且 `output_item.done` 的
/// reasoning item 摘要同样带 marker——只存完成态的客户端靠它回带；
/// encrypted_content 仍是覆盖上游原文的干净凭据。
#[tokio::test]
async fn e2e_session_marker_stream_responses_client() {
    let (base_url, _captured) = spawn_mock_chat_thinking_capture().await;
    let (state, _db) = setup_with("openai-chat", base_url, "deepseek-x", "test-model").await;

    let body = json!({
        "model": "test-model",
        "input": "run ls",
        "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::OpenAiResponse, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let sse = String::from_utf8_lossy(&bytes).to_string();

    let tag = sse_tag(&sse).expect("responses 入口的 reasoning 流也应带水印: {sse}");
    let frames: Vec<Value> = sse
        .lines()
        .filter_map(|l| l.strip_prefix("data:"))
        .filter_map(|d| serde_json::from_str::<Value>(d.trim()).ok())
        .collect();
    // 首个推理摘要增量即 marker 首部
    assert!(
        frames.iter().any(|f| {
            f["type"] == "response.reasoning_summary_text.delta"
                && f["delta"].as_str() == Some(&format!(" [mb:{tag}]"))
        }),
        "marker 应随摘要增量下发: {sse}"
    );
    // done item 的 summary 同样带 marker（只存完成态的客户端靠它回带）
    let item_done = frames
        .iter()
        .find(|f| f["type"] == "response.output_item.done" && f["item"]["type"] == "reasoning")
        .expect("应有 reasoning 的 output_item.done");
    assert_eq!(
        item_done["item"]["summary"][0]["text"].as_str().unwrap(),
        format!(" [mb:{tag}]think hard"),
        "done item 摘要应带 marker 首部: {item_done}"
    );
    assert_eq!(
        item_done["item"]["encrypted_content"].as_str().unwrap(),
        "chat:think hard",
        "凭据仍覆盖上游原文、不含 marker: {item_done}"
    );
}

/// 流式对照组：整流无推理块（正文 + tool_use）⇒ 不打标记，正文与工具块照常。
#[tokio::test]
async fn e2e_session_marker_absent_in_stream_without_reasoning() {
    let base_url = spawn_mock_sse(sse_text_then_tool_use()).await;
    let (state, _db) = setup_with("anthropic", base_url, "claude-x", "test-model").await;

    let body = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }], "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let sse = String::from_utf8_lossy(&bytes).to_string();
    assert!(!sse.contains("mb:"), "无推理块的流不得打标记: {sse}");
    assert!(sse.contains("Visible"), "正文仍要送达");
    assert!(sse.contains("dropme"), "工具块仍要完整转发");
}

/// 流式对照组（anthropic 上游）：thinking + tool_use 轮照样打标记。
#[tokio::test]
async fn e2e_session_marker_present_in_stream_tool_use_round() {
    let base_url = spawn_mock_sse(sse_thinking_then_tool_use()).await;
    let (state, _db) = setup_with("anthropic", base_url, "claude-x", "test-model").await;

    let body = json!({
        "model": "test-model", "max_tokens": 64,
        "messages": [{ "role": "user", "content": "Hi" }], "stream": true
    });
    let resp = dispatch::handle_request(state, Protocol::Anthropic, body, vec![], None)
        .await
        .expect("dispatch 应成功");
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let sse = String::from_utf8_lossy(&bytes).to_string();

    let tag = sse_tag(&sse).expect("tool_use 轮的 thinking 流应带水印: {sse}");
    let frames = anthropic_frames(&sse);
    let td = thinking_deltas(&frames, 0);
    assert_eq!(
        td.first().map(String::as_str),
        Some(format!(" [mb:{tag}]").as_str()),
        "marker 应是 thinking 首部增量: {td:?}"
    );
    assert!(
        frames
            .iter()
            .any(|f| f["content_block"]["type"] == "tool_use"),
        "工具块完整转发: {sse}"
    );
}
