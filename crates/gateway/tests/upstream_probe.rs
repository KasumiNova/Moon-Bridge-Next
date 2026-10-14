//! 真实网络探针：验证 reqwest 连接池复用对 cctq 这类中转的行为。
//! 手动跑：`CCTQ_KEY=sk-xxx cargo test -p moonbridge-gateway --test upstream_probe -- --ignored --nocapture`

use futures::StreamExt;
use moonbridge_core::Protocol;
use moonbridge_gateway::upstream;
use moonbridge_gateway::GatewayConfig;
use moonbridge_protocol::UpstreamRequest;
use serde_json::Value;
use std::time::Instant;

fn make_req(body: Value) -> UpstreamRequest {
    UpstreamRequest {
        method: http::Method::POST,
        url: "https://www.cctq.ai/v1/responses".into(),
        headers: vec![
            ("content-type".into(), "application/json".into()),
            (
                "authorization".into(),
                format!(
                    "Bearer {}",
                    std::env::var("CCTQ_KEY").expect("CCTQ_KEY 未设置")
                ),
            ),
        ],
        body,
        stream: true,
    }
}

fn tiny_stream_body() -> Value {
    serde_json::json!({
        "model": "gpt-6.1-sol",
        "input": "用一句话解释什么是TCP",
        "stream": true,
        "max_output_tokens": 512
    })
}

/// 在同一个 reqwest::Client（即同一连接池）上连续发两个流式请求，
/// 第二次故意闲置 60s 后再发——若中转/LB 提前静默关闭空闲连接，
/// 第二发会命中半死连接（发送失败或流中途挂起）。
#[tokio::test]
#[ignore]
async fn probe_pooled_connection_reuse() {
    let client = upstream::build_client(&GatewayConfig::default()).unwrap();
    let body = std::fs::read_to_string("/tmp/replay_req.json")
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .unwrap_or_else(tiny_stream_body);

    for (i, wait) in [(1usize, 0u64), (2, 60)] {
        if wait > 0 {
            println!("--- 第 {i} 发前先闲置 {wait}s ---");
            tokio::time::sleep(std::time::Duration::from_secs(wait)).await;
        }
        let t = Instant::now();
        match upstream::send(&client, &make_req(body.clone())).await {
            Err(e) => println!("req{i}: 发送失败 {e}"),
            Ok(r) => {
                println!("req{i}: HTTP {}", r.status());
                let mut chunks = Box::pin(upstream::response_to_chunks(
                    r,
                    Protocol::OpenAiResponse,
                    Some("probe".into()),
                ));
                let mut n = 0usize;
                let mut last = Instant::now();
                let mut tail = String::new();
                while let Some(item) = chunks.next().await {
                    match item {
                        Ok(c) => {
                            n += 1;
                            last = Instant::now();
                            let ev = c.event.clone().unwrap_or_default();
                            let data = serde_json::to_string(&c.data).unwrap_or_default();
                            if ev.contains("completed")
                                || ev.contains("failed")
                                || data.contains("\"failed\"")
                            {
                                tail =
                                    format!("{ev} {}", data.chars().take(200).collect::<String>());
                            }
                        }
                        Err(e) => {
                            println!("req{i}: chunk 错误（第 {n} 个后）: {e}");
                            break;
                        }
                    }
                }
                println!(
                    "req{i}: {n} 个 chunk，总耗时 {:?}，末chunk距今 {:?}，终帧: {}",
                    t.elapsed(),
                    last.elapsed(),
                    if tail.is_empty() { "无" } else { &tail }
                );
            }
        }
    }
}
