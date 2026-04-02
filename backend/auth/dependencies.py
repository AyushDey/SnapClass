# auth/dependencies.py

from fastapi import Request, HTTPException
from .jwt_handler import verify_token


def get_current_user(request: Request):
    token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = verify_token(token)

        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")

        return payload

    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")