//! Static web assets embedded from `public/` at build time, plus the Axum
//! fallback handler that serves them.
//!
//! `build.rs` walks the root-level `public/` directory and generates
//! `OUT_DIR/public_assets.rs` — an [`Asset`] struct and a [`lookup`] function
//! over forward-slash relative paths, each arm an `include_bytes!` of the
//! source-tree file. The whole tree is baked into the binary, so `seeker serve`
//! needs no filesystem access at runtime.

include!(concat!(env!("OUT_DIR"), "/public_assets.rs"));

use axum::body::Body;
use axum::http::{HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};

/// Axum `.fallback` handler: serve an embedded static asset when no built-in
/// route matched the request path. Only `GET`/`HEAD` reach assets; every other
/// method and every miss returns a bare 404.
pub async fn handler(method: Method, uri: Uri) -> Response {
    if method != Method::GET && method != Method::HEAD {
        return (StatusCode::NOT_FOUND, "Not Found").into_response();
    }

    let key = normalize_path(uri.path());
    match lookup(&key) {
        Some(asset) => {
            tracing::debug!(path = %key, "static asset served");
            // For HEAD we send the headers without a body. (Hyper also strips
            // the body on HEAD at the transport layer, but an explicit empty
            // body avoids any content-length ambiguity.)
            let body = if method == Method::HEAD {
                Body::empty()
            } else {
                Body::from(asset.bytes)
            };
            Response::builder()
                .status(StatusCode::OK)
                .header(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static(asset.content_type),
                )
                .body(body)
                // `content_type` is a `&'static str` from a build-time table and
                // valid by construction, so this never errors.
                .expect("static asset response is always valid")
        }
        None => {
            tracing::debug!(path = %key, "static asset miss");
            (StatusCode::NOT_FOUND, "Not Found").into_response()
        }
    }
}

/// Map a request URI path to a [`lookup`] key: strip the leading `/`, and treat
/// the root and any directory-style path as `index.html`.
///
/// Path-safety invariant: `lookup` is an exact-match in-memory table that never
/// touches the filesystem, so a `..` (or percent-encoded) traversal simply
/// fails to match and 404s — there is no path-join sink to exploit. We
/// therefore key on the raw, undecoded path and do not special-case `..`. If a
/// future version embeds assets whose names need percent-decoding, decode once
/// here AND reintroduce an explicit `..`-segment reject before keeping this
/// invariant.
fn normalize_path(path: &str) -> String {
    let trimmed = path.strip_prefix('/').unwrap_or(path);
    if trimmed.is_empty() {
        "index.html".to_string()
    } else if trimmed.ends_with('/') {
        format!("{trimmed}index.html")
    } else {
        trimmed.to_string()
    }
}
