from fastapi import APIRouter, Response, HTTPException, Depends
from sqlalchemy.orm import Session

from .schemas import LoginRequest, RegisterRequest, TokenResponse
from .jwt_handler import create_access_token, create_refresh_token
from .password_handler import verify_password, hash_password

from db import sessionLocal
from db_actions import DBActions

router = APIRouter()


def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/register", response_model=TokenResponse)
def register(data: RegisterRequest, db: Session = Depends(get_db)):

    db_actions = DBActions(db)

    existing_user = db_actions.get_user_by_email(data.email)

    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    hashed_password = hash_password(data.password)

    db_actions.create_user(data.email, hashed_password)
    db_actions.commit()

    return {"message": "User registered successfully"}



@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest, response: Response, db: Session = Depends(get_db)):

    db_actions = DBActions(db)

    user = db_actions.get_user_by_email(data.email)

    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token({"sub": user.email})
    refresh_token = create_refresh_token({"sub": user.email})

    response.set_cookie("access_token", access_token, httponly=True,secure=True)
    response.set_cookie("refresh_token", refresh_token, httponly=True,secure=True)

    return {"message": "Login successful"}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")

    return {"message": "Logged out successfully"}