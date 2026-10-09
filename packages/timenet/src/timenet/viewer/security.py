"""Authentication and browser-request checks for the local viewer."""

from __future__ import annotations

from collections.abc import Callable
import hmac
import logging
import secrets
from typing import Any

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def new_token() -> str:
    """Return an unguessable process-local bearer token."""
    return secrets.token_urlsafe(48)


def require_request(request: Request, token: str, authority: str) -> None:
    """Reject requests that are not same-origin authenticated viewer API calls.

    Raises:
        HTTPException: If the host, origin, or bearer token is invalid.
    """
    if request.headers.getlist("host") != [authority]:
        raise HTTPException(status_code=403, detail="invalid host")
    origin = request.headers.get("origin")
    if origin is not None and origin != f"http://{authority}":
        raise HTTPException(status_code=403, detail="invalid origin")
    if request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}:
        raise HTTPException(status_code=403, detail="cross-site request denied")
    scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
    if scheme != "Bearer" or not hmac.compare_digest(supplied, token):
        raise HTTPException(status_code=401, detail="authentication required")


_MAX_REQUEST_BYTES = 65536


class RequestBoundary:
    """Authenticate before parsing and enforce bounded request and response bodies."""

    def __init__(self, app: ASGIApp, *, token: str, authority: str) -> None:
        self.app, self.token, self.authority = app, token, authority

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Apply the authenticated bounded ASGI boundary."""  # noqa: DOC501 - errors become HTTP responses
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        try:
            if request.headers.getlist("host") != [self.authority]:
                raise HTTPException(403, "invalid host")
            if scope["path"].startswith("/api/"):
                require_request(request, self.token, self.authority)
            chunks = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                chunks.extend(message.get("body", b""))
                if len(chunks) > _MAX_REQUEST_BYTES:
                    raise HTTPException(413, "request exceeds 64 KiB")
                if not message.get("more_body", False):
                    break
        except HTTPException as exc:
            await JSONResponse({"message": exc.detail}, status_code=exc.status_code)(scope, receive, send)
            return
        consumed = False

        async def bounded_receive() -> Message:
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": bytes(chunks), "more_body": False}

        messages: list[Message] = []
        size = 0

        async def bounded_send(message: Message) -> None:  # noqa: RUF029 - ASGI send contract
            nonlocal size
            size += len(message.get("body", b""))
            if size <= 8 * 2**20:
                messages.append(message)

        await self.app(scope, bounded_receive, bounded_send)
        if size > 8 * 2**20:
            await JSONResponse({"message": "response exceeds 8 MiB; narrow the selection"}, status_code=413)(
                scope, receive, send
            )
        else:
            for message in messages:
                await send(message)


async def security_headers(request: Request, call_next: Callable) -> Any:
    """Return a protected response, logging sanitized unexpected HTTP failures."""
    try:
        response = await call_next(request)
    except Exception:
        # Do not include request URLs, headers, or payloads in the log context.
        logging.getLogger(__name__).exception("Unexpected inspection request failure")
        response = JSONResponse({"message": "inspection failed"}, status_code=500)
    response.headers.update(
        {
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Cross-Origin-Resource-Policy": "same-origin",
            "Content-Security-Policy": (
                "default-src 'none'; script-src 'self'; connect-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; font-src 'self'; object-src 'none'; base-uri 'none'; "
                "frame-ancestors 'none'; form-action 'none'"
            ),
        }
    )
    return response
