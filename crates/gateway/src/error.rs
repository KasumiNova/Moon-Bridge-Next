use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use moonbridge_core::ErrorBody;

/// 网关错误。可直接转换为 HTTP 响应（OpenAI 风格错误体）。
#[derive(Debug, thiserror::Error)]
pub enum GatewayError {
    #[error("协议错误: {0}")]
    Protocol(#[from] moonbridge_core::CoreError),

    #[error("存储错误: {0}")]
    Store(#[from] moonbridge_store::StoreError),

    #[error("HTTP 错误: {0}")]
    Http(#[from] reqwest::Error),

    #[error("上游错误[{status}]: {message}")]
    Upstream { status: u16, message: String },

    #[error("路由错误: {0}")]
    Route(String),

    #[error("认证失败: {0}")]
    Auth(String),

    #[error("{0}")]
    Other(String),
}

impl GatewayError {
    pub fn status_code(&self) -> StatusCode {
        match self {
            GatewayError::Auth(_) => StatusCode::UNAUTHORIZED,
            GatewayError::Route(_) => StatusCode::NOT_FOUND,
            GatewayError::Upstream { status, .. } => {
                StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY)
            }
            GatewayError::Protocol(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = ErrorBody::new("gateway_error", self.to_string());
        (status, axum::Json(body)).into_response()
    }
}

pub type Result<T> = std::result::Result<T, GatewayError>;
