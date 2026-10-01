"""Small, fixed user hints for known provider failures; never echo error text."""

import httpx
from openai import APIConnectionError


def provider_error_hint(error: Exception) -> str | None:
    """Read only structured status/code fields, not messages, URLs or headers.

    OpenAI-compatible SDKs expose either the full response JSON or the inner
    ``error`` object as ``body``. Unknown bodies/codes deliberately get no hint
    so callers can retain their existing generic error message.
    """
    body = getattr(error, "body", None)
    code = None
    if isinstance(body, dict):
        nested = body.get("error")
        code = nested.get("code") if isinstance(nested, dict) else body.get("code")
    code = code.casefold() if isinstance(code, str) else None
    status = getattr(error, "status_code", None)

    if code == "arrearage":
        return "模型服务账户欠费或额度状态异常，请检查账户后重试。"
    if code in {"invalidapikey", "invalid_api_key"} or status == 401:
        return "模型服务身份验证失败，请检查 API 密钥和访问权限后重试。"
    if status == 429:
        return "模型服务请求受限，请稍后重试并检查调用额度。"
    if isinstance(error, (TimeoutError, ConnectionError, APIConnectionError, httpx.TransportError)):
        return "模型服务连接失败或超时，请检查网络后重试。"
    return None
