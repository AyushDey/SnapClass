# auth/password_handler.py

from passlib.context import CryptContext
from passlib.hash import pbkdf2_sha256

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    # 👇 convert to fixed-length hash (removes 72 byte limit)
    hashed_password = pbkdf2_sha256.hash(password)
    return hashed_password


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pbkdf2_sha256.verify(plain_password, hashed_password)