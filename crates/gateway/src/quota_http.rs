use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use moonbridge_plugin::{HttpRequest, HttpResponse};
use serde_json::Value;
use url::{Host, Url};

const HTTP_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const OFFICIAL_ORIGINS: &[&str] = &[
    "https://api.deepseek.com",
    "https://api.moonshot.cn",
    "https://api.moonshot.ai",
    "https://api.siliconflow.cn",
    "https://api.siliconflow.com",
    "https://api.anthropic.com",
    "https://open.bigmodel.cn",
    "https://openrouter.ai",
    "https://api.kimi.com",
    "https://api.commandcode.ai",
    "https://api.z.ai",
];

#[derive(Clone, Debug, Default)]
pub struct QuotaNetworkPolicy {
    private_origins: HashSet<String>,
    denied_reason: Option<String>,
}

impl QuotaNetworkPolicy {
    pub fn from_environment(egress_proxy: Option<String>) -> Self {
        let configured = match std::env::var("MOONBRIDGE_QUOTA_PRIVATE_ORIGINS") {
            Ok(value) => serde_json::from_str::<Vec<String>>(&value).map_err(|_| {
                "MOONBRIDGE_QUOTA_PRIVATE_ORIGINS 必须是 origin 字符串的 JSON 数组".to_string()
            }),
            Err(std::env::VarError::NotPresent) => Ok(Vec::new()),
            Err(_) => Err("MOONBRIDGE_QUOTA_PRIVATE_ORIGINS 编码无效".to_string()),
        };
        match configured.and_then(|origins| Self::from_config(origins, egress_proxy)) {
            Ok(policy) => policy,
            Err(reason) => Self {
                denied_reason: Some(reason),
                ..Self::default()
            },
        }
    }

    pub fn from_config(
        private_origins: Vec<String>,
        egress_proxy: Option<String>,
    ) -> Result<Self, String> {
        if egress_proxy.is_some() {
            return Err("配额查询不支持出站代理：无法安全校验代理侧 DNS".to_string());
        }
        let mut allowed = HashSet::new();
        for origin in private_origins {
            let url = parse_url(&origin)?;
            if url.path() != "/" || url.query().is_some() || url.fragment().is_some() {
                return Err(
                    "配额私网白名单必须是精确的 HTTP(S) origin，不能包含路径、查询或片段"
                        .to_string(),
                );
            }
            if let Some(ip) = literal_ip(&url) {
                validate_ip(ip, true)?;
            }
            allowed.insert(url.origin().ascii_serialization());
        }
        Ok(Self {
            private_origins: allowed,
            denied_reason: None,
        })
    }

    /// 策略是否因配置问题被整体禁用（配置了出站代理，或私网白名单非法）。
    /// 一旦禁用，[`Self::request`] 会以该原因拒绝每一次查询——宿主应在启动时检查
    /// 并告警，否则配额看板会"静默全红"，只能从逐卡错误里反推根因。
    pub fn denied_reason(&self) -> Option<&str> {
        self.denied_reason.as_deref()
    }

    pub(crate) async fn request(
        &self,
        base_url: &str,
        req: HttpRequest,
    ) -> Result<HttpResponse, String> {
        if let Some(reason) = &self.denied_reason {
            return Err(reason.clone());
        }
        let timeout = Duration::from_millis(
            req.timeout_ms
                .unwrap_or(HTTP_TIMEOUT.as_millis() as u64)
                .clamp(1, HTTP_TIMEOUT.as_millis() as u64),
        );
        tokio::time::timeout(timeout, self.request_inner(base_url, req))
            .await
            .map_err(|_| "配额 HTTP 请求超时".to_string())?
    }

    async fn request_inner(
        &self,
        base_url: &str,
        req: HttpRequest,
    ) -> Result<HttpResponse, String> {
        let url = parse_url(&req.url)?;
        let origin = url.origin().ascii_serialization();
        let private_allowed = self.private_origins.contains(&origin);
        let same_origin = if base_url.trim().is_empty() {
            OFFICIAL_ORIGINS.contains(&origin.as_str())
        } else {
            parse_url(base_url.trim())?.origin() == url.origin()
        };
        if !same_origin && !private_allowed {
            return Err("配额 HTTP 请求不在卡片同源或明确授权的 origin 内".to_string());
        }
        let mut headers = reqwest::header::HeaderMap::new();
        for (name, value) in &req.headers {
            let name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| "配额 HTTP 请求头名称无效".to_string())?;
            if name == reqwest::header::HOST || name.as_str().starts_with("proxy-") {
                return Err("配额 HTTP 请求不能覆写 Host 或代理请求头".to_string());
            }
            let value = reqwest::header::HeaderValue::from_str(value)
                .map_err(|_| "配额 HTTP 请求头值无效".to_string())?;
            headers.append(name, value);
        }
        let method = reqwest::Method::from_bytes(req.method.to_uppercase().as_bytes())
            .map_err(|_| "配额 HTTP 方法无效".to_string())?;
        if method == reqwest::Method::CONNECT {
            return Err("配额 HTTP 请求不支持 CONNECT".to_string());
        }
        let port = url.port_or_known_default().ok_or("配额 HTTP 端口无效")?;
        let host = match url.host() {
            Some(Host::Domain(host)) => host.to_string(),
            Some(Host::Ipv4(ip)) => ip.to_string(),
            Some(Host::Ipv6(ip)) => ip.to_string(),
            None => return Err("配额 HTTP 主机无效".to_string()),
        };
        let addresses: Vec<SocketAddr> = if let Some(ip) = literal_ip(&url) {
            vec![SocketAddr::new(ip, port)]
        } else {
            tokio::net::lookup_host((host.as_str(), port))
                .await
                .map_err(|_| "配额 HTTP DNS 解析失败".to_string())?
                .collect()
        };
        if addresses.is_empty() {
            return Err("配额 HTTP DNS 解析未返回地址".to_string());
        }
        for address in &addresses {
            validate_ip(address.ip(), private_allowed)?;
        }
        let client = reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(HTTP_TIMEOUT)
            .timeout(HTTP_TIMEOUT)
            .resolve_to_addrs(&host, &addresses)
            .build()
            .map_err(|_| "配额 HTTP 安全客户端创建失败".to_string())?;
        let mut builder = client.request(method, url).headers(headers);
        if let Some(body) = &req.body {
            builder = builder.json(body);
        }
        let mut response = builder
            .send()
            .await
            .map_err(|_| "配额 HTTP 请求失败".to_string())?;
        if response.status().is_redirection() {
            return Err("配额 HTTP 请求禁止重定向".to_string());
        }
        if response
            .headers()
            .get_all(reqwest::header::CONTENT_ENCODING)
            .iter()
            .any(|value| {
                value
                    .to_str()
                    .map_or(true, |value| !value.trim().eq_ignore_ascii_case("identity"))
            })
        {
            return Err("配额 HTTP 响应使用不支持的压缩编码".to_string());
        }
        let status = response.status().as_u16();
        let headers = response
            .headers()
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_str().unwrap_or("").to_string()))
            .collect();
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| "配额 HTTP 响应读取失败".to_string())?
        {
            if chunk.len() > MAX_RESPONSE_BYTES.saturating_sub(bytes.len()) {
                return Err("配额 HTTP 响应超过 1 MiB 上限".to_string());
            }
            bytes.extend_from_slice(&chunk);
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let body = serde_json::from_str(&text).unwrap_or(Value::String(text));
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }
}

fn parse_url(raw: &str) -> Result<Url, String> {
    let mut url = Url::parse(raw).map_err(|_| "配额 HTTP URL 无效".to_string())?;
    let authority = raw
        .split_once("://")
        .map(|(_, rest)| rest.split(['/', '?', '#', '\\']).next().unwrap_or(""))
        .unwrap_or("");
    if !matches!(url.scheme(), "http" | "https")
        || raw.trim() != raw
        || raw.chars().any(|c| c.is_ascii_control() || c == '\\')
        || authority.is_empty()
        || url.host().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || authority.contains('@')
    {
        return Err("配额 HTTP URL 必须使用 HTTP(S) 且不能包含用户信息".to_string());
    }
    if let Some(Host::Domain(domain)) = url.host() {
        let domain = domain.trim_end_matches('.').to_ascii_lowercase();
        if domain.is_empty() || domain.contains('*') {
            return Err("配额 HTTP 主机无效".to_string());
        }
        if matches!(
            domain.as_str(),
            "metadata.google.internal"
                | "metadata.goog"
                | "instance-data.ec2.internal"
                | "metadata.azure.internal"
        ) {
            return Err("配额 HTTP 请求禁止访问元数据服务".to_string());
        }
        url.set_host(Some(&domain))
            .map_err(|_| "配额 HTTP 主机无效".to_string())?;
    }
    Ok(url)
}

fn literal_ip(url: &Url) -> Option<IpAddr> {
    match url.host()? {
        Host::Ipv4(ip) => Some(IpAddr::V4(ip)),
        Host::Ipv6(ip) => Some(IpAddr::V6(ip)),
        Host::Domain(_) => None,
    }
}

fn validate_ip(ip: IpAddr, private_allowed: bool) -> Result<(), String> {
    let (public, private) = match ip {
        IpAddr::V4(ip) => {
            let [a, b, c, d] = ip.octets();
            if (a == 169 && b == 254)
                || [a, b, c, d] == [100, 100, 100, 200]
                || [a, b, c, d] == [168, 63, 129, 16]
                || [a, b, c, d] == [192, 0, 0, 192]
            {
                return Err("配额 HTTP 请求禁止访问元数据服务".to_string());
            }
            let private =
                ip.is_private() || ip.is_loopback() || (a == 100 && (64..=127).contains(&b));
            (is_public_v4(ip), private)
        }
        IpAddr::V6(ip) => {
            if ip.to_ipv4_mapped().is_some()
                || ip == Ipv6Addr::new(0xfd00, 0xec2, 0, 0, 0, 0, 0, 0x254)
            {
                return Err("配额 HTTP 请求禁止 IPv4 映射地址或元数据服务".to_string());
            }
            let segments = ip.segments();
            let private = ip.is_loopback() || (segments[0] & 0xfe00 == 0xfc00);
            let public = (segments[0] & 0xe000 == 0x2000)
                && !(segments[0] == 0x2001 && segments[1] < 0x0200)
                && !(segments[0] == 0x2001 && segments[1] == 0x0db8)
                && segments[0] != 0x2002
                && !(segments[0] == 0x3fff && segments[1] < 0x1000);
            (public, private)
        }
    };
    if public || (private_allowed && private) {
        Ok(())
    } else {
        Err("配额 HTTP 请求禁止访问私网或保留地址".to_string())
    }
}

fn is_public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(ip.is_private()
        || ip.is_loopback()
        || a == 0
        || a >= 224
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 192 && b == 88 && c == 99)
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::get;
    use axum::{Json, Router};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn request(url: &str) -> HttpRequest {
        HttpRequest {
            method: "GET".into(),
            url: url.into(),
            headers: vec![("Authorization".into(), "Bearer secret-token".into())],
            body: None,
            timeout_ms: Some(2000),
        }
    }

    async fn serve(app: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (origin, task)
    }

    #[test]
    fn urls_reject_unsafe_schemes_credentials_and_metadata() {
        for raw in [
            "file:///etc/passwd",
            "ftp://example.com",
            "http://user:secret-token@example.com",
            "http://@example.com",
            " http://example.com",
            "http://example.com\n",
            "http://example.com\\@127.0.0.1",
            "http://metadata.google.internal",
            "http://METADATA.GOOGLE.INTERNAL.",
            "http://metadata.goog",
            "http://instance-data.ec2.internal",
            "http://metadata.azure.internal",
        ] {
            let error = parse_url(raw).unwrap_err();
            assert!(!error.contains("secret-token"), "{error}");
        }
        assert_eq!(
            parse_url("https://EXAMPLE.COM.:443/path")
                .unwrap()
                .origin()
                .ascii_serialization(),
            "https://example.com"
        );
    }

    #[test]
    fn private_and_reserved_ips_are_denied_without_exact_authorization() {
        for ip in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "100.64.0.1",
            "::1",
            "fd12::1",
        ] {
            let ip = ip.parse().unwrap();
            assert!(validate_ip(ip, false).is_err(), "{ip}");
            assert!(validate_ip(ip, true).is_ok(), "{ip}");
        }
        for ip in [
            "0.0.0.0",
            "169.254.169.254",
            "100.100.100.200",
            "168.63.129.16",
            "192.0.0.192",
            "192.0.2.1",
            "198.18.0.1",
            "198.51.100.1",
            "203.0.113.1",
            "224.0.0.1",
            "255.255.255.255",
            "::",
            "fe80::1",
            "ff02::1",
            "::ffff:127.0.0.1",
            "::ffff:8.8.8.8",
            "fd00:ec2::254",
            "2001:db8::1",
            "2002:0808:0808::1",
            "2001::1",
            "3fff::1",
        ] {
            let ip = ip.parse().unwrap();
            assert!(validate_ip(ip, false).is_err(), "{ip}");
            assert!(
                validate_ip(ip, true).is_err(),
                "whitelist must not allow {ip}"
            );
        }
        for ip in ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"] {
            assert!(validate_ip(ip.parse().unwrap(), false).is_ok(), "{ip}");
        }
        for raw in ["http://2130706433", "http://0x7f000001", "http://127.1"] {
            let ip = literal_ip(&parse_url(raw).unwrap()).unwrap();
            assert!(validate_ip(ip, false).is_err(), "{raw}");
        }
    }

    #[test]
    fn configuration_requires_exact_origins_and_rejects_proxies() {
        for origin in [
            "http://127.0.0.1/path",
            "http://127.0.0.1?query",
            "http://127.0.0.1#fragment",
            "http://*.example.com",
            "http://169.254.169.254",
            "http://[::ffff:127.0.0.1]",
        ] {
            assert!(
                QuotaNetworkPolicy::from_config(vec![origin.into()], None).is_err(),
                "{origin}"
            );
        }
        let error = QuotaNetworkPolicy::from_config(
            Vec::new(),
            Some("http://secret-token@proxy.example".into()),
        )
        .unwrap_err();
        assert!(error.contains("代理"));
        assert!(!error.contains("secret-token"));
    }

    #[tokio::test]
    async fn origin_mismatch_private_dns_and_denied_environment_fail_closed() {
        let policy = QuotaNetworkPolicy::default();
        for url in [
            "https://other.example/path",
            "http://example.com/path",
            "https://example.com:444/path",
        ] {
            let error = policy
                .request("https://example.com", request(url))
                .await
                .unwrap_err();
            assert!(error.contains("origin"), "{error}");
        }
        for url in ["http://127.0.0.1:9", "http://localhost:9", "http://[::1]:9"] {
            let error = policy.request(url, request(url)).await.unwrap_err();
            assert!(error.contains("私网"), "{url}: {error}");
        }
        let denied =
            QuotaNetworkPolicy::from_environment(Some("http://secret-token@proxy.example".into()));
        let error = denied
            .request("http://127.0.0.1", request("http://127.0.0.1"))
            .await
            .unwrap_err();
        assert!(!error.contains("secret-token"));
        assert!(!error.is_empty());
    }

    #[tokio::test]
    async fn exact_private_origin_allows_roundtrip_but_not_other_ports_or_hosts() {
        let hits = Arc::new(AtomicUsize::new(0));
        let handler_hits = hits.clone();
        let app = Router::new().route(
            "/",
            get(move |headers: axum::http::HeaderMap| {
                let hits = handler_hits.clone();
                async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    assert_eq!(headers.get("authorization").unwrap(), "Bearer secret-token");
                    Json(json!({"left": 0}))
                }
            }),
        );
        let (origin, server) = serve(app).await;
        let policy = QuotaNetworkPolicy::from_config(vec![origin.clone()], None).unwrap();
        let response = policy.request(&origin, request(&origin)).await.unwrap();
        assert_eq!(response.status, 200);
        assert_eq!(response.body, json!({"left": 0}));
        for target in [
            "http://127.0.0.1:9".to_string(),
            origin.replace("127.0.0.1", "localhost"),
        ] {
            assert!(policy.request(&origin, request(&target)).await.is_err());
        }
        for name in ["Host", "hOsT", "Proxy-Authorization", "proxy-connection"] {
            let mut req = request(&origin);
            req.headers.push((name.into(), "secret-token".into()));
            let error = policy.request(&origin, req).await.unwrap_err();
            assert!(error.contains("请求头"), "{error}");
            assert!(!error.contains("secret-token"));
        }
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        server.abort();
    }

    #[tokio::test]
    async fn redirects_are_not_followed_even_to_an_authorized_origin() {
        let hits = Arc::new(AtomicUsize::new(0));
        let target_hits = hits.clone();
        let (target, target_server) = serve(Router::new().route(
            "/",
            get(move || {
                let hits = target_hits.clone();
                async move {
                    hits.fetch_add(1, Ordering::SeqCst);
                    "unexpected"
                }
            }),
        ))
        .await;
        let location = target.clone();
        let (origin, server) = serve(Router::new().route(
            "/",
            get(move || {
                let location = location.clone();
                async move { (axum::http::StatusCode::FOUND, [("location", location)], "") }
            }),
        ))
        .await;
        let policy = QuotaNetworkPolicy::from_config(vec![origin.clone(), target], None).unwrap();
        let error = policy.request(&origin, request(&origin)).await.unwrap_err();
        assert!(error.contains("重定向"), "{error}");
        assert_eq!(hits.load(Ordering::SeqCst), 0);
        server.abort();
        target_server.abort();
    }

    #[tokio::test]
    async fn response_size_timeout_and_errors_are_bounded() {
        let app = Router::new()
            .route("/limit", get(|| async { "x".repeat(MAX_RESPONSE_BYTES) }))
            .route(
                "/oversize",
                get(|| async { "x".repeat(MAX_RESPONSE_BYTES + 1) }),
            )
            .route(
                "/slow",
                get(|| async {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    "late"
                }),
            );
        let (origin, server) = serve(app).await;
        let policy = QuotaNetworkPolicy::from_config(vec![origin.clone()], None).unwrap();
        let response = policy
            .request(&origin, request(&format!("{origin}/limit")))
            .await
            .unwrap();
        assert_eq!(response.body.as_str().unwrap().len(), MAX_RESPONSE_BYTES);
        let error = policy
            .request(
                &origin,
                request(&format!("{origin}/oversize?token=secret-token")),
            )
            .await
            .unwrap_err();
        assert!(error.contains("1 MiB"), "{error}");
        assert!(!error.contains("secret-token"));
        let mut req = request(&format!("{origin}/slow?token=secret-token"));
        req.timeout_ms = Some(10);
        let error = policy.request(&origin, req).await.unwrap_err();
        assert!(error.contains("超时"), "{error}");
        assert!(!error.contains("secret-token"));
        server.abort();
    }
}
