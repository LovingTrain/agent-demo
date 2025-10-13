# backend/api/auth_api.py
from fastapi import APIRouter, HTTPException, Body
from service.auth_service import (
    LoginIn,
    TokenOut,
    RegisterIn,
    verify_password,
    create_access_token,
    get_user_by_username,
    hash_password,
    persist_user,
    USERS,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenOut)
def login(body: LoginIn = Body(...)):
    u = get_user_by_username(body.username)
    if not u or not verify_password(body.password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(int(u["id"]))
    return TokenOut(access_token=token)


@router.post("/register")
def register(body: RegisterIn = Body(...)):
    if get_user_by_username(body.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    new_id = max([int(v.get("id", 0)) for v in USERS.values()] + [0]) + 1
    pw_hash = hash_password(body.password)
    persist_user(body.username, new_id, pw_hash)
    return {"id": new_id, "username": body.username}
