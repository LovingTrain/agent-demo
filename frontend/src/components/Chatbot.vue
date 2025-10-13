<template>
  <div class="chat-container">
    <Sidebar
      :current-session-id="currentSessionId"
      :sessions="sessions"
      :is-collapsed="sidebarCollapsed"
      :is-mobile="isMobile"
      @change-session="handleChangeSession"
      @create-new-session="handleCreateNewSession"
      @delete-session="handleDeleteSession"
      @rename-session="handleRenameSession"
      @duplicate-session="handleDuplicateSession"
      @toggle-sidebar="toggleSidebar"
      @close-sidebar="closeSidebar"
    />

    <div class="main-content">
      <ChatHeader
        :current-session-id="currentSessionId"
        :sidebar-collapsed="sidebarCollapsed"
        :is-mobile="isMobile"
        @toggle-sidebar="toggleSidebar"
      />

      <ChatMessages
        ref="chatMessagesRef"
        :messages="messages"
        :is-loading="isLoading"
        :current-streaming-message="currentStreamingMessage"
      />

      <ChatInput
        :is-loading="isLoading"
        @send-message="handleSendMessage"
        @clear-messages="handleClearMessages"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, watch, computed, type Ref } from 'vue'
import Sidebar from './Sidebar.vue'
import ChatHeader from './ChatHeader.vue'
import ChatMessages from './ChatMessages.vue'
import ChatInput from './ChatInput.vue'
import type { Message, Session } from '@/types/chat'
import 'highlight.js/styles/atom-one-dark.css'

/**
 * 配置与鉴权
 */
const API_BASE = import.meta.env.VITE_API_BASE || ''
const getToken = () => localStorage.getItem('token') || ''

async function api<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init.headers || {}),
      'Authorization': `Bearer ${getToken()}`
    }
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(text || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

/**
 * 组件状态
 */
const messages = reactive<Message[]>([])
const sessions = reactive<Session[]>([])
const isLoading = ref<boolean>(false)
const currentSessionId = ref<string>('')
const currentStreamingMessage = ref<Message | null>(null)
const chatMessagesRef: Ref<any> = ref(null)
const sidebarCollapsed = ref<boolean>(false)
const currentStreamAborter = ref<AbortController | null>(null)

const isMobile = computed(() => {
  if (typeof window === 'undefined') return false
  return window.innerWidth <= 768
})

/**
 * 工具函数
 */
const generateId = (): string => Date.now().toString(36) + Math.random().toString(36).slice(2)
const generateDefaultSessionId = (): string => 'chat_' + Date.now().toString(36)

const scrollToBottom = (): void => {
  if (chatMessagesRef.value) {
    chatMessagesRef.value.scrollToBottom()
  }
}

const addMessage = (text: string, type: 'user' | 'ai', isStreaming = false): Message => {
  const message: Message = {
    id: generateId(),
    text,
    type,
    timestamp: new Date(),
    isStreaming
  }
  messages.push(message)
  scrollToBottom()
  return message
}

function replaceMessagesFromServer(rows: Array<{ id: number; role: string; content: string; created_at: string }>) {
  messages.splice(0)
  for (const r of rows) {
    messages.push({
      id: String(r.id),
      text: r.content,
      type: r.role === 'user' ? 'user' : 'ai',
      timestamp: new Date(r.created_at),
      isStreaming: false
    })
  }
  scrollToBottom()
}

/**
 * 后端 API 封装：会话与消息
 */
async function ensureSession(sessionId: string, name?: string) {
  await api('/sessions', { method: 'POST', body: JSON.stringify({ id: sessionId, name: name || sessionId }) })
}

async function fetchSessions() {
  const data = await api<Array<{ id: string; name: string; created_at: string; updated_at: string }>>('/sessions')
  sessions.splice(0)
  sessions.push(
    ...data.map((s) => ({
      id: s.id,
      name: s.name,
      createdAt: new Date(s.created_at),
      lastUsed: new Date(s.updated_at)
    }))
  )
}

async function fetchMessages(sessionId: string) {
  const data = await api<Array<{ id: number; role: string; content: string; created_at: string }>>(
    `/sessions/${encodeURIComponent(sessionId)}/messages`
  )
  replaceMessagesFromServer(data)
}

async function renameSessionApi(sessionId: string, newName: string) {
  await api(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    body: JSON.stringify({ name: newName })
  })
}

async function deleteSessionApi(sessionId: string) {
  await api(`/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' })
}

async function addUserMessageApi(sessionId: string, content: string) {
  await api(`/sessions/${encodeURIComponent(sessionId)}/messages/user`, {
    method: 'POST',
    body: JSON.stringify({ content })
  })
}

/**
 * 侧边栏控制
 */
const toggleSidebar = (): void => {
  sidebarCollapsed.value = !sidebarCollapsed.value
  localStorage.setItem('sidebar_collapsed', sidebarCollapsed.value.toString())
}
const closeSidebar = (): void => { if (isMobile.value) sidebarCollapsed.value = true }

/**
 * 会话操作
 */
const handleCreateNewSession = async (sessionId: string): Promise<void> => {
  // 后端创建
  await ensureSession(sessionId, sessionId)

  // 前端状态
  currentSessionId.value = sessionId
  await fetchSessions()
  await fetchMessages(sessionId)

  // 记住当前会话
  localStorage.setItem('current_session_id', sessionId)
}

const handleChangeSession = async (sessionId: string): Promise<void> => {
  if (sessionId === currentSessionId.value) return

  // 中断可能存在的流
  currentStreamAborter.value?.abort()
  currentStreamingMessage.value = null

  currentSessionId.value = sessionId
  localStorage.setItem('current_session_id', sessionId)

  await fetchMessages(sessionId)
  await fetchSessions() // 让排序更新
}

const handleDeleteSession = async (sessionId: string): Promise<void> => {
  await deleteSessionApi(sessionId)
  // 如果删的是当前会话，选择新的
  await fetchSessions()
  if (sessions.length > 0) {
    const nextId = sessions[0].id
    currentSessionId.value = nextId
    localStorage.setItem('current_session_id', nextId)
    await fetchMessages(nextId)
  } else {
    const defaultId = generateDefaultSessionId()
    await handleCreateNewSession(defaultId)
  }
}

const handleRenameSession = async (sessionId: string, newName: string): Promise<void> => {
  await renameSessionApi(sessionId, newName)
  await fetchSessions()
  if (currentSessionId.value === sessionId) {
    currentSessionId.value = newName
    localStorage.setItem('current_session_id', newName)
    await fetchMessages(newName)
  }
}

const handleDuplicateSession = async (sessionId: string): Promise<void> => {
  // 前端生成新ID，创建一个空会话即可（如需复制消息，可在后端实现复制接口）
  const newId = `${sessionId}_copy_${Date.now().toString(36)}`
  await ensureSession(newId, newId)
  await fetchSessions()
  await handleChangeSession(newId)
}

/**
 * 初始化
 */
onMounted(async () => {
  // 侧边栏状态
  const savedCollapsed = localStorage.getItem('sidebar_collapsed')
  sidebarCollapsed.value = savedCollapsed !== null ? savedCollapsed === 'true' : isMobile.value

  // 加载会话
  await fetchSessions()

  // 当前会话：优先 localStorage 记忆；否则第一个；没有则创建默认
  const savedSessionId = localStorage.getItem('current_session_id')
  if (savedSessionId && sessions.some(s => s.id === savedSessionId)) {
    currentSessionId.value = savedSessionId
  } else if (sessions.length > 0) {
    currentSessionId.value = sessions[0].id
  } else {
    const defaultId = generateDefaultSessionId()
    await handleCreateNewSession(defaultId)
    return
  }

  await fetchMessages(currentSessionId.value)
})

/**
 * 自动刷新会话排序（当 messages 变化时可选择刷新 sessions 的 updated_at 视图）
 */
watch(() => currentSessionId.value, async () => {
  // 切换会话后刷新列表顺序
  await fetchSessions()
})

/**
 * 清空当前会话消息（仅前端展示上清空；真实删除请做一个 DELETE /sessions/{id}/messages 接口再调用）
 */
const handleClearMessages = (): void => {
  messages.splice(0)
}

/**
 * SSE 解析
 */
const parseSseBuffer = (buf: string, onData: (d: string) => void) => {
  let rest = buf
  let idx = rest.indexOf('\n\n')
  while (idx >= 0) {
    const evt = rest.slice(0, idx).trim()
    rest = rest.slice(idx + 2)
    if (evt.startsWith('data:')) {
      onData(evt.slice(5).trimStart())
    }
    idx = rest.indexOf('\n\n')
  }
  return rest
}

/**
 * 发送消息（对接 SQL API 与 SSE）
 */
const handleSendMessage = async (userMessage: string): Promise<void> => {
  const token = getToken()
  if (!token) {
    alert('请先登录或设置 API Key')
    return
  }
  if (!currentSessionId.value) {
    const defaultId = generateDefaultSessionId()
    await handleCreateNewSession(defaultId)
  }

  // UI：先追加用户消息
  addMessage(userMessage, 'user')

  // 中断上一次流
  currentStreamAborter.value?.abort()
  currentStreamAborter.value = new AbortController()
  const signal = currentStreamAborter.value.signal

  isLoading.value = true
  try {
    // 确保会话存在并落库用户消息
    await ensureSession(currentSessionId.value, currentSessionId.value)
    await addUserMessageApi(currentSessionId.value, userMessage)

    // 发起流式推理
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({ input: userMessage, session_id: currentSessionId.value }),
      signal
    })
    if (!res.ok) throw new Error(await res.text().catch(() => `HTTP ${res.status}`))

    const aiMsg = addMessage('', 'ai', true)
    currentStreamingMessage.value = aiMsg

    const reader = res.body?.getReader()
    if (!reader) throw new Error('Response body is not readable')

    const decoder = new TextDecoder('utf-8')
    let buf = ''
    let aiText = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      buf = parseSseBuffer(buf, (chunk) => {
        aiText += chunk
        const idx = messages.findIndex(m => m.id === aiMsg.id)
        if (idx !== -1) {
          messages[idx].text = aiText
          scrollToBottom()
        }
      })
    }

    // 标记完成
    const idx = messages.findIndex(m => m.id === aiMsg.id)
    if (idx !== -1) messages[idx].isStreaming = false
    currentStreamingMessage.value = null

    // 可选：再拉一次服务端消息，确保与数据库一致
    // await fetchMessages(currentSessionId.value)
    await fetchSessions() // 更新会话 lastUsed 排序
  } catch (e) {
    console.error('发送消息失败:', e)
    if (currentStreamingMessage.value) {
      const i = messages.findIndex(m => m.id === currentStreamingMessage.value!.id)
      if (i > -1) {
        messages[i].text = '抱歉，发送消息时出现错误，请稍后重试。'
        messages[i].isStreaming = false
      }
    }
    currentStreamingMessage.value = null
  } finally {
    isLoading.value = false
    currentStreamAborter.value = null
  }
}
</script>

<style scoped>
.chat-container {
  display: flex;
  height: 100vh;
  background: #0a0a0a;
  color: #e5e5e5;
  font-family: 'SF Pro Display', system-ui, -apple-system, sans-serif;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

@media (max-width: 768px) {
  .chat-container {
    position: relative;
  }
}
</style>