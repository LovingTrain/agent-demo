# service/auth_service.py
import os
import toml
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from fastapi import Header, HTTPException
from pydantic import BaseModel, Field

from repo.base import SessionLocal
from repo import user_repo

AUTH_TOML_PATH = os.getenv("AUTH_TOML_PATH", "config/auth.toml")


def _load_auth_config() -> Dict[str, Any]:
    if not os.path.exists(AUTH_TOML_PATH):
        raise FileNotFoundError(f"Auth config not found: {AUTH_TOML_PATH}")
    return toml.load(AUTH_TOML_PATH)


cfg_auth = _load_auth_config()

# 不再使用 users.toml；仅保留配置文件中的 jwt 与 api_keys
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


# ------------- ORM 读写：替换原 users.toml 路径 -------------
def get_user_by_id(uid: int) -> Optional[Dict[str, Any]]:
    db = SessionLocal()
    try:
        u = user_repo.get_by_id(db, uid)
        if not u:
            return None
        return {"id": u.id, "username": u.username, "password_hash": u.password_hash}
    finally:
        db.close()


def get_user_by_username(uname: str) -> Optional[Dict[str, Any]]:
    db = SessionLocal()
    try:
        u = user_repo.get_by_username(db, uname)
        if not u:
            return None
        return {"id": u.id, "username": u.username, "password_hash": u.password_hash}
    finally:
        db.close()


def create_user(username: str, password: str) -> Dict[str, Any]:
    db = SessionLocal()
    try:
        if user_repo.get_by_username(db, username):
            raise ValueError("用户名已存在")
        pw_hash = hash_password(password)
        u = user_repo.create_user(db, username, pw_hash)
        db.commit()
        return {"id": u.id, "username": u.username, "password_hash": u.password_hash}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# API Key 直通：如命中配置的 API Key，则发放一个“虚拟/共享用户”
# 为保持外键完整性，这里确保存在一个用户记录。
def try_auth_with_api_key(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    if _hash_api_key(token) not in API_KEY_HASHES:
        return None

    default_uid = cfg_auth.get("default_api_user_id")  # 可选：如果配置了固定 id
    db = SessionLocal()
    try:
        # 优先使用固定 id；否则按固定用户名创建/复用
        if default_uid is not None:
            u = user_repo.get_by_id(db, int(default_uid))
            if not u:
                # 创建固定 id 的用户（需要你的 ORM 允许显式设置 id）
                # 如果 crud_user.create_user 不支持指定 id，可单独写一段插入逻辑
                from repo.models import User  # 局部导入，避免循环

                u = User(
                    id=int(default_uid), username="api_key_user", password_hash="!"
                )
                db.add(u)
                db.commit()
            return {"username": u.username, "id": u.id}

        # 未指定固定 id，则按用户名复用
        uname = "api_key_user"
        u = user_repo.get_by_username(db, uname)
        if not u:
            from repo.models import User

            u = User(username=uname, password_hash="!")
            db.add(u)
            db.commit()
        return {"username": u.username, "id": u.id}
    finally:
        db.close()


# ------------- FastAPI 依赖与 DTO -------------
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

    # 先尝试 JWT
    uid = decode_access_token(token)
    if uid is not None:
        u = get_user_by_id(uid)
        if not u:
            raise HTTPException(status_code=401, detail="User not found")
        return CurrentUser(id=int(u["id"]), username=u["username"])

    # 再尝试 API Key
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


# 兼容原接口：保留 persist_user 名称，但实现改为写入 SQLite
def persist_user(username: str, user_id: int, password_hash_str: str):
    # 保持函数签名不变，内部改为 upsert 到表
    db = SessionLocal()
    try:
        from repo.models import User

        u = user_repo.get_by_id(db, user_id)
        if u:
            u.username = username
            u.password_hash = password_hash_str
        else:
            db.add(User(id=user_id, username=username, password_hash=password_hash_str))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
