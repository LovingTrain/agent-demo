// src/utils/auth.ts
import { apiFetch, setToken } from "./api";

export interface LoginReq {
  username: string;
  password: string;
}

export interface TokenOut {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
}

export async function login(req: LoginReq): Promise<void> {
  const data = await apiFetch<TokenOut>("/auth/login", {
    method: "POST",
    body: JSON.stringify(req),
  });
  setToken(data.access_token);
}

export async function register(req: LoginReq): Promise<void> {
  await apiFetch("/auth/register", {
    method: "POST",
    body: JSON.stringify(req),
  });
}