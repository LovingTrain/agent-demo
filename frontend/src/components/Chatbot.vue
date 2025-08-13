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
import type { Message, ChatRequest, Session } from '@/types/chat'
import Sidebar from './Sidebar.vue'
import ChatHeader from './ChatHeader.vue'
import ChatMessages from './ChatMessages.vue'
import ChatInput from './ChatInput.vue'
import 'highlight.js/styles/atom-one-dark.css'

const messages = reactive<Message[]>([])
const sessions = reactive<Session[]>([])
const isLoading = ref<boolean>(false)
const currentSessionId = ref<string>('')
const currentStreamingMessage = ref<Message | null>(null)
const chatMessagesRef: Ref<any> = ref(null)
const sidebarCollapsed = ref<boolean>(false)

// 检测移动端
const isMobile = computed(() => {
  if (typeof window === 'undefined') return false
  return window.innerWidth <= 768
})

// 生成唯一ID
const generateId = (): string => {
  return Date.now().toString(36) + Math.random().toString(36).substr(2)
}

// 生成默认会话ID
const generateDefaultSessionId = (): string => {
  return 'chat_' + Date.now().toString(36)
}

// 侧边栏控制
const toggleSidebar = (): void => {
  sidebarCollapsed.value = !sidebarCollapsed.value
  localStorage.setItem('sidebar_collapsed', sidebarCollapsed.value.toString())
}

const closeSidebar = (): void => {
  if (isMobile.value) {
    sidebarCollapsed.value = true
  }
}

// 创建新会话
const handleCreateNewSession = (sessionId: string): void => {
  if (sessions.some(s => s.id === sessionId)) {
    console.error('会话ID已存在')
    return
  }
  
  saveCurrentSession()
  
  currentSessionId.value = sessionId
  messages.splice(0)
  
  const newSession: Session = {
    id: sessionId,
    name: sessionId,
    createdAt: new Date(),
    lastUsed: new Date()
  }
  
  sessions.unshift(newSession)
  localStorage.setItem('chat_sessions', JSON.stringify(sessions))
}

// 切换会话
const handleChangeSession = (sessionId: string): void => {
  if (sessionId === currentSessionId.value) return
  
  saveCurrentSession()
  currentSessionId.value = sessionId
  messages.splice(0)
  loadSessionMessages(sessionId)
  updateSessionLastUsed(sessionId)
}

// 删除会话
const handleDeleteSession = (sessionId: string): void => {
  const index = sessions.findIndex(s => s.id === sessionId)
  if (index > -1) {
    sessions.splice(index, 1)
    localStorage.removeItem(`chat_messages_${sessionId}`)
    
    if (sessionId === currentSessionId.value) {
      if (sessions.length > 0) {
        handleChangeSession(sessions[0].id)
      } else {
        const defaultSessionId = generateDefaultSessionId()
        handleCreateNewSession(defaultSessionId)
      }
    }
    
    localStorage.setItem('chat_sessions', JSON.stringify(sessions))
  }
}

// 重命名会话
const handleRenameSession = (sessionId: string, newName: string): void => {
  const sessionIndex = sessions.findIndex(s => s.id === sessionId)
  if (sessionIndex > -1) {
    // 检查新名称是否已存在
    if (sessions.some(s => s.id === newName && s.id !== sessionId)) {
      alert('会话名称已存在')
      return
    }
    
    const oldSessionId = sessions[sessionIndex].id
    
    // 更新会话信息
    sessions[sessionIndex].id = newName
    sessions[sessionIndex].name = newName
    
    // 更新存储
    const oldMessages = localStorage.getItem(`chat_messages_${oldSessionId}`)
    if (oldMessages) {
      localStorage.setItem(`chat_messages_${newName}`, oldMessages)
      localStorage.removeItem(`chat_messages_${oldSessionId}`)
    }
    
    // 如果是当前会话，更新当前会话ID
    if (currentSessionId.value === oldSessionId) {
      currentSessionId.value = newName
    }
    
    localStorage.setItem('chat_sessions', JSON.stringify(sessions))
  }
}

// 复制会话
const handleDuplicateSession = (sessionId: string): void => {
  const session = sessions.find(s => s.id === sessionId)
  if (!session) return
  
  const newSessionId = `${sessionId}_copy_${Date.now().toString(36)}`
  const savedMessages = localStorage.getItem(`chat_messages_${sessionId}`)
  
  // 创建新会话
  const newSession: Session = {
    id: newSessionId,
    name: newSessionId,
    createdAt: new Date(),
    lastUsed: new Date()
  }
  
  sessions.unshift(newSession)
  
  // 复制消息
  if (savedMessages) {
    localStorage.setItem(`chat_messages_${newSessionId}`, savedMessages)
  }
  
  localStorage.setItem('chat_sessions', JSON.stringify(sessions))
  
  // 切换到新会话
  handleChangeSession(newSessionId)
}

// 更新会话最后使用时间
const updateSessionLastUsed = (sessionId: string): void => {
  const sessionIndex = sessions.findIndex(s => s.id === sessionId)
  if (sessionIndex > -1) {
    sessions[sessionIndex].lastUsed = new Date()
    const session = sessions.splice(sessionIndex, 1)[0]
    sessions.unshift(session)
    localStorage.setItem('chat_sessions', JSON.stringify(sessions))
  }
}

// 保存当前会话
const saveCurrentSession = (): void => {
  if (!currentSessionId.value) return
  
  localStorage.setItem(`chat_messages_${currentSessionId.value}`, JSON.stringify(messages))
  
  const existingIndex = sessions.findIndex(s => s.id === currentSessionId.value)
  if (existingIndex > -1) {
    sessions[existingIndex].lastUsed = new Date()
    const session = sessions.splice(existingIndex, 1)[0]
    sessions.unshift(session)
  } else if (messages.length > 0) {
    const sessionData: Session = {
      id: currentSessionId.value,
      name: currentSessionId.value,
      createdAt: new Date(),
      lastUsed: new Date()
    }
    sessions.unshift(sessionData)
  }
  
  localStorage.setItem('chat_sessions', JSON.stringify(sessions))
}

// 加载会话消息
const loadSessionMessages = (sessionId: string): void => {
  const savedMessages = localStorage.getItem(`chat_messages_${sessionId}`)
  if (savedMessages) {
    try {
      const parsedMessages = JSON.parse(savedMessages)
      messages.splice(0, messages.length, ...parsedMessages.map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp),
        isStreaming: false
      })))
      scrollToBottom()
    } catch (error) {
      console.error('Failed to load session messages:', error)
    }
  }
}

// 初始化
onMounted(() => {
  // 加载侧边栏状态
  const savedCollapsed = localStorage.getItem('sidebar_collapsed')
  if (savedCollapsed !== null) {
    sidebarCollapsed.value = savedCollapsed === 'true'
  } else {
    // 移动端默认收起
    sidebarCollapsed.value = isMobile.value
  }
  
  // 加载会话列表
  const savedSessions = localStorage.getItem('chat_sessions')
  if (savedSessions) {
    try {
      const parsedSessions = JSON.parse(savedSessions)
      sessions.splice(0, sessions.length, ...parsedSessions.map((session: any) => ({
        ...session,
        createdAt: new Date(session.createdAt),
        lastUsed: new Date(session.lastUsed)
      })))
      sessions.sort((a, b) => b.lastUsed.getTime() - a.lastUsed.getTime())
    } catch (error) {
      console.error('Failed to load sessions:', error)
    }
  }
  
  // 设置当前会话
  if (sessions.length > 0) {
    currentSessionId.value = sessions[0].id
    loadSessionMessages(sessions[0].id)
  } else {
    const defaultSessionId = generateDefaultSessionId()
    handleCreateNewSession(defaultSessionId)
  }
})

// 监听消息变化并自动保存
let saveTimeout: any = null
watch(messages, () => {
  if (currentSessionId.value && messages.length > 0) {
    clearTimeout(saveTimeout)
    saveTimeout = setTimeout(() => {
      saveCurrentSession()
    }, 1000)
  }
}, { deep: true })

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

const handleClearMessages = (): void => {
  messages.splice(0)
  localStorage.removeItem(`chat_messages_${currentSessionId.value}`)
}

const handleSendMessage = async (userMessage: string): Promise<void> => {
  // 添加用户消息
  addMessage(userMessage, 'user')
  
  isLoading.value = true
  
  try {
    const requestBody: ChatRequest = {
      input: userMessage,
      session_id: currentSessionId.value
    }

    const response = await fetch('/chat/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    // 创建AI回复消息
    const aiMessage = addMessage('', 'ai', true)
    currentStreamingMessage.value = aiMessage
    
    const reader = response.body?.getReader()
    if (!reader) {
      throw new Error('Response body is not readable')
    }

    const decoder = new TextDecoder()
    let allData = ''

    // 读取所有数据
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      allData += decoder.decode(value, { stream: true })
    }

    console.log('接收到的完整数据:', allData)

    // 提取内容
    let content = allData
    if (content.startsWith('data: ')) {
      content = content.substring(6).trim()
    }

    console.log('提取的内容:', content)
    console.log('内容长度:', content.length)

    // 确保 aiMessage 还在 messages 数组中
    const messageIndex = messages.findIndex(m => m.id === aiMessage.id)
    if (messageIndex === -1) {
      console.error('Message not found in array')
      return
    }

    // 直接在 messages 数组中更新，模拟打字效果
    messages[messageIndex].text = ''
    
    for (let i = 0; i <= content.length; i++) {
      messages[messageIndex].text = content.substring(0, i)
      scrollToBottom()
      if (i < content.length) {
        await new Promise(resolve => setTimeout(resolve, 20))
      }
    }
    
    // 完成
    messages[messageIndex].isStreaming = false
    currentStreamingMessage.value = null
    
  } catch (error) {
    console.error('发送消息失败:', error)
    if (currentStreamingMessage.value) {
      const messageIndex = messages.findIndex(m => m.id === currentStreamingMessage.value!.id)
      if (messageIndex > -1) {
        messages[messageIndex].text = '抱歉，发送消息时出现错误，请稍后重试。'
        messages[messageIndex].isStreaming = false
      }
    }
    currentStreamingMessage.value = null
  } finally {
    isLoading.value = false
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