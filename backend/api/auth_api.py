# api/auth_api.py
from fastapi import APIRouter, HTTPException, Body
from service.auth_service import (
    LoginIn,
    TokenOut,
    RegisterIn,
    verify_password,
    create_access_token,
    hash_password,
)
from repo.base import SessionLocal, migrate
from repo import user_repo

router = APIRouter(prefix="/auth", tags=["auth"])

# 确保表存在（也可以在应用启动处统一调用）
migrate()


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn = Body(...)):
    db = SessionLocal()
    try:
        u = user_repo.get_by_username(db, body.username)
        if not u or not verify_password(body.password, u.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_access_token(int(u.id))
        return TokenOut(access_token=token)
    finally:
        db.close()


@router.post("/register")
def register(body: RegisterIn = Body(...)):
    db = SessionLocal()
    try:
        if user_repo.get_by_username(db, body.username):
            raise HTTPException(status_code=400, detail="Username already exists")
        pw_hash = hash_password(body.password)
        u = user_repo.create_user(db, body.username, pw_hash)
        db.commit()
        return {"id": int(u.id), "username": u.username}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
