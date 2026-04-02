# middleware/auth_middleware.py

from fastapi import Request
from fastapi.responses import JSONResponse
from auth.jwt_handler import verify_token
from auth.config import PUBLIC_ROUTES


async def auth_middleware(request: Request, call_next):
    path = request.url.path

    if path not in PUBLIC_ROUTES:
        token = request.cookies.get("access_token")

        if not token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"}
            )

        try:
            verify_token(token)
        except Exception:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or expired token"}
            )

    response = await call_next(request)
    return response