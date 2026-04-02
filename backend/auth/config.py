
SECRET_KEY = "super-secret-key-change-this"
ALGORITHM = "HS256"

ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7


PUBLIC_ROUTES = [
    "/docs",
    "/openapi.json",
    "/auth/login",
    "/auth/register",
    "/auth/refresh"
]


PROTECTED_ROUTES = [
    "/predict",
    "/classify",
]