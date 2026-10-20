"""Request boundary for the viewer's browser API."""

from collections.abc import Awaitable, Callable
import hmac
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from timenet.errors import TimeFValidationError


def normalize_origin(origin: str) -> str:
    """Return an HTTP(S) origin with browser-normalized host and port.

    Returns:
        The origin with a lowercase host and without a default port.

    Raises:
        TimeFValidationError: If the value is not an HTTP(S) origin.
    """
    try:
        parsed = urlsplit(origin)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise TimeFValidationError("viewer origin must be a valid HTTP(S) origin") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or any((parsed.username is not None, parsed.password is not None, parsed.path, parsed.query, parsed.fragment))
    ):
        raise TimeFValidationError("viewer origin must be an HTTP(S) origin without a path or credentials")
    try:
        host = f"[{host}]" if ":" in host else host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise TimeFValidationError("viewer origin must have a valid hostname") from exc
    default_port = 443 if parsed.scheme == "https" else 80
    authority = host if port in {None, default_port} else f"{host}:{port}"
    return f"{parsed.scheme}://{authority}"


def install_security(app: FastAPI, *, token: str, origin: str) -> None:
    """Require same-origin bearer requests and add browser security headers.

    Args:
        app: Viewer application to protect.
        token: Process-local secret printed in the launch URL fragment.
        origin: Exact browser origin allowed to reach this application.

    Raises:
        TimeFValidationError: If the configured origin is not an HTTP(S) origin.
    """  # noqa: DOC502 - origin validation is delegated to normalize_origin
    origin = normalize_origin(origin)
    authority = urlsplit(origin).netloc

    @app.middleware("http")
    async def boundary(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Reject foreign browser requests before routing them.

        Returns:
            The authorized response or a small denial response.
        """
        if request.headers.get("host") != authority:
            response: Response = JSONResponse({"message": "invalid host"}, status_code=403)
        elif request.headers.get("origin") not in {None, origin}:
            response = JSONResponse({"message": "invalid origin"}, status_code=403)
        elif request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}:
            response = JSONResponse({"message": "cross-site request denied"}, status_code=403)
        elif request.url.path.startswith("/api/") and not _authorized(request, token):
            response = JSONResponse({"message": "authentication required"}, status_code=401)
        else:
            response = await call_next(request)
        response.headers.update(
            {
                "Cache-Control": "no-store",
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Cross-Origin-Resource-Policy": "same-origin",
                "Content-Security-Policy": (
                    "default-src 'none'; script-src 'self'; connect-src 'self'; "
                    "style-src 'self'; img-src 'self' data:; base-uri 'none'; "
                    "frame-ancestors 'none'; form-action 'none'"
                ),
            }
        )
        return response


def _authorized(request: Request, token: str) -> bool:
    scheme, separator, supplied = request.headers.get("authorization", "").partition(" ")
    # Compare bytes: compare_digest rejects non-ASCII str, and headers decode as Latin-1.
    return separator == " " and scheme.lower() == "bearer" and hmac.compare_digest(supplied.encode(), token.encode())
