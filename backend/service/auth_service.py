# backend/service/auth_service.py
import os
import toml
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

AUTH_TOML_PATH = os.getenv("AUTH_TOML_PATH", "config/auth.toml")


def _load_auth_config() -> Dict[str, Any]:
    if not os.path.exists(AUTH_TOML_PATH):
        raise FileNotFoundError(f"Auth config not found: {AUTH_TOML_PATH}")
    return toml.load(AUTH_TOML_PATH)


cfg_auth = _load_auth_config()

USERS_FILE = cfg_auth.get("users_file", "config/users.toml")
JWT_SECRET = cfg_auth.get("jwt_secret", os.getenv("CHAT_JWT_SECRET", "replace-me"))
JWT_ALG = "HS256"
JWT_EXPIRE_MIN = int(cfg_auth.get("jwt_expire_minutes", 60))

RAW_API_KEYS: List[str] = list(cfg_auth.get("api_keys", []))
ENV_API_KEYS = os.getenv("CHAT_API_KEYS")
if ENV_API_KEYS:
    RAW_API_KEYS.extend([x.strip() for x in ENV_API_KEYS.split(",") if x.strip()])


def _hash_api_key(k: str) -> str:
    import hashlib as _h

    return _h.sha256(k.encode()).hexdigest()


API_KEY_HASHES = {_hash_api_key(k) for k in RAW_API_KEYS if k}


def _load_users() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(USERS_FILE):
        return {}
    data = toml.load(USERS_FILE)
    return data.get("users", {})


USERS = _load_users()


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, pw_hash: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), pw_hash.encode())
    except Exception:
        return False


def create_access_token(user_id: int) -> str:
    now = datetime.utcnow()
    payload = {
        "sub": str(user_id),
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MIN),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def decode_access_token(token: str) -> Optional[int]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        if payload.get("type") != "access":
            return None
        return int(payload["sub"])
    except Exception:
        return None


def get_user_by_id(uid: int) -> Optional[Dict[str, Any]]:
    for uname, data in USERS.items():
        if int(data.get("id")) == uid:
            return {"username": uname, **data}
    return None


def get_user_by_username(uname: str) -> Optional[Dict[str, Any]]:
    data = USERS.get(uname)
    if not data:
        return None
    return {"username": uname, **data}


def try_auth_with_api_key(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    if _hash_api_key(token) in API_KEY_HASHES:
        default_uid = int(cfg_auth.get("default_api_user_id", 0))
        return {"username": "api_key_user", "id": default_uid}
    return None


class CurrentUser(BaseModel):
    id: int
    username: str


async def get_current_user(authorization: Optional[str] = Header(None)) -> CurrentUser:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    try:
        scheme, token = authorization.split(" ", 1)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Unsupported auth scheme")

    uid = decode_access_token(token)
    if uid is not None:
        u = get_user_by_id(uid)
        if not u:
            raise HTTPException(status_code=401, detail="User not found")
        return CurrentUser(id=int(u["id"]), username=u["username"])

    u = try_auth_with_api_key(token)
    if u:
        return CurrentUser(id=int(u["id"]), username=u["username"])

    raise HTTPException(status_code=401, detail="Invalid token or API key")


class LoginIn(BaseModel):
    username: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRE_MIN * 60


class RegisterIn(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)


def persist_user(username: str, user_id: int, password_hash_str: str):
    data = {"users": USERS}
    USERS[username] = {"id": user_id, "password_hash": password_hash_str}
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        toml.dump(data, f)
