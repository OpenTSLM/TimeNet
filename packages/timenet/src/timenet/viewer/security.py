"""Request boundary for the viewer's browser API."""

from collections.abc import Awaitable, Callable
import hmac
from urllib.parse import urlsplit

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse


def install_security(app: FastAPI, *, token: str, origin: str) -> None:
    """Require same-origin bearer requests and add browser security headers.

    Args:
        app: Viewer application to protect.
        token: Process-local secret printed in the launch URL fragment.
        origin: Exact browser origin allowed to reach this application, as the launcher builds
            it (``http://127.0.0.1:<port>``).
    """
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
