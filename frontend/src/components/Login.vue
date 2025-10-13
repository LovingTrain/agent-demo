<template>
  <div class="auth-wrap">
    <div class="card">
      <div class="brand">
        <div class="logo">🤖</div>
        <div class="title">AI Chat 登录</div>
        <div class="sub">欢迎回来，请先登录以继续使用</div>
      </div>

      <div class="tabs">
        <button
          class="tab"
          :class="{ active: activeTab === 'account' }"
          @click="activeTab = 'account'"
        >
          账号登录
        </button>
        <button
          class="tab"
          :class="{ active: activeTab === 'apikey' }"
          @click="activeTab = 'apikey'"
        >
          使用 API Key
        </button>
      </div>

      <!-- 账号登录 -->
      <form v-if="activeTab === 'account'" class="form" @submit.prevent="onLogin">
        <div class="field">
          <label>用户名</label>
          <input
            v-model.trim="username"
            placeholder="输入用户名"
            autocomplete="username"
            :disabled="loading"
          />
        </div>
        <div class="field">
          <label>密码</label>
          <div class="pwd">
            <input
              v-model="password"
              :type="showPwd ? 'text' : 'password'"
              placeholder="输入密码"
              autocomplete="current-password"
              :disabled="loading"
            />
            <button type="button" class="eye" @click="showPwd = !showPwd" :title="showPwd ? '隐藏' : '显示'">
              <svg v-if="!showPwd" width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M2 12s3.8-6 10-6 10 6 10 6-3.8 6-10 6-10-6-10-6Z" stroke="currentColor" stroke-width="1.6" opacity=".9"/>
                <circle cx="12" cy="12" r="3.2" stroke="currentColor" stroke-width="1.6"/>
              </svg>
              <svg v-else width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M3 3l18 18" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
                <path d="M2 12s3.8-6 10-6c2.2 0 4 .6 5.5 1.5M22 12s-3.8 6-10 6c-2.2 0-4-.6-5.5-1.5" stroke="currentColor" stroke-width="1.6" opacity=".9"/>
                <circle cx="12" cy="12" r="3.2" stroke="currentColor" stroke-width="1.6" opacity=".9"/>
              </svg>
            </button>
          </div>
        </div>

        <div class="actions">
          <button class="btn primary" type="submit" :disabled="loading || !canLogin">
            <span v-if="!loading">登录</span>
            <span v-else class="spinner"></span>
          </button>
          <RouterLink class="btn ghost link" to="/register">去注册</RouterLink>
        </div>

        <p v-if="error" class="error">{{ error }}</p>
      </form>

      <!-- API Key 登录 -->
      <form v-else class="form" @submit.prevent="useApiKey">
        <div class="field">
          <label>API Key</label>
          <input
            v-model.trim="apiKey"
            placeholder="粘贴后端颁发的 API Key"
            :disabled="loading"
          />
        </div>
        <div class="hint">
          - 使用方式：直接将 API Key 作为 Bearer 令牌使用。<br />
          - 建议仅在内网或开发环境使用。
        </div>

        <div class="actions single">
          <button class="btn primary" type="submit" :disabled="loading || !apiKey">
            <span v-if="!loading">使用 API Key</span>
            <span v-else class="spinner"></span>
          </button>
        </div>

        <p v-if="error" class="error">{{ error }}</p>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { RouterLink } from 'vue-router'

const API_BASE = import.meta.env.VITE_API_BASE || ''
const activeTab = ref<'account' | 'apikey'>('account')
const username = ref('')
const password = ref('')
const apiKey = ref('')
const showPwd = ref(false)
const loading = ref(false)
const error = ref('')

const canLogin = computed(() => username.value.length >= 3 && password.value.length >= 6)

async function onLogin() {
  error.value = ''
  loading.value = true
  try {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: username.value, password: password.value })
    })
    if (!res.ok) throw new Error(await safeText(res))
    const data = await res.json()
    localStorage.setItem('token', data.access_token)
    window.location.replace('/')
  } catch (e: any) {
    error.value = e?.message || '登录失败，请稍后重试'
  } finally {
    loading.value = false
  }
}

function useApiKey() {
  error.value = ''
  if (!apiKey.value) {
    error.value = '请输入 API Key'
    return
  }
  localStorage.setItem('token', apiKey.value)
  window.location.replace('/')
}

async function safeText(res: Response) {
  try { return await res.text() } catch { return `HTTP ${res.status}` }
}
</script>

<style scoped>
/* 最初版风格：深色背景 + 玻璃卡片，使用 padding/margin 控制留白 */
.auth-wrap {
  min-height: 100vh;
  background:
    radial-gradient(1200px 600px at 20% 0%, rgba(59,130,246,.15), transparent 60%),
    radial-gradient(1200px 600px at 80% 100%, rgba(34,197,94,.12), transparent 60%),
    #0a0a0a;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  color: #e5e5e5;
}

.card {
  width: 100%;
  max-width: 480px;
  background: rgba(20, 20, 20, 0.78);
  backdrop-filter: blur(10px);
  border: 1px solid #242424;
  box-shadow: 0 10px 30px rgba(0,0,0,.35);
  border-radius: 16px;
  padding: 28px 32px;
}

.brand {
  text-align: center;
  margin-bottom: 12px;
}
.logo {
  width: 56px; height: 56px;
  margin: 0 auto 8px;
  background: #1f2937;
  border-radius: 14px;
  display: grid; place-items: center;
  font-size: 28px;
  border: 1px solid #2b3340;
}
.title { font-size: 28px; font-weight: 800; }
.sub { color: #a0a0a0; font-size: 14px; margin-top: 6px; }

.tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin: 18px 0 12px;
}
.tab {
  height: 48px;
  padding: 0 12px;
  background: #151515;
  color: #cbd5e1;
  border: 1px solid #2a2a2a;
  border-radius: 12px;
  cursor: pointer;
  transition: all .2s ease;
}
.tab.active {
  background: #1f2937;
  color: #fff;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59,130,246,.12) inset;
}
.tab:hover { background: #1a1a1a; }

.form {
  display: grid;
  gap: 14px;
  margin-top: 10px;
}

.field label {
  display: block;
  font-size: 12px;
  color: #a3a3a3;
  margin-bottom: 6px;
}

.field input {
  width: 95%;
  height: 48px;
  padding: 0 14px;
  background: #0b0b0b;
  color: #e5e5e5;
  border: 1px solid #2a2a2a;
  border-radius: 12px;
  outline: none;
  transition: border-color .2s, box-shadow .2s;
}
.field input::placeholder { color: #6b7280; }
.field input:focus {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59,130,246,.12);
}

.pwd { position: relative; }
.eye {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  width: 32px; height: 32px;
  display: grid; place-items: center;
  border-radius: 8px;
  border: 1px solid transparent;
  background: transparent;
  color: #cbd5e1;
  cursor: pointer;
  transition: background .2s, border-color .2s, color .2s;
}
.eye:hover { background: #141414; border-color: #2a2a2a; color: #ffffff; }

.hint {
  font-size: 12px;
  color: #9ca3af;
  line-height: 1.6;
  background: #0e0e0e;
  border: 1px solid #222;
  padding: 10px 12px;
  border-radius: 10px;
}

.actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-top: 4px;
}
.actions.single { grid-template-columns: 1fr; }

.btn {
  height: 48px;
  border-radius: 12px;
  cursor: pointer;
  border: 1px solid transparent;
  transition: transform .02s, background .2s, border-color .2s, color .2s, box-shadow .2s;
  font-weight: 700;
  display: grid; place-items: center;
}
.btn:active { transform: translateY(1px); }

.btn.primary {
  background: #2563eb;
  color: #fff;
  border-color: #1d4ed8;
}
.btn.primary:hover { background: #1d4ed8; }

.btn.ghost {
  background: #151515;
  color: #e5e5e5;
  border-color: #2a2a2a;
}
.btn.ghost:hover { background: #1a1a1a; }
.btn.link { text-decoration: none; }

.error {
  margin-top: -4px;
  color: #f87171;
  font-size: 13px;
}

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid rgba(255,255,255,.25);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin .8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

@media (max-width: 640px) {
  .card { padding: 20px; }
  .title { font-size: 22px; }
}
</style>