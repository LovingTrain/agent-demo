export interface Message {
  text: string
  type: 'user' | 'ai'
  timestamp: Date
  isStreaming?: boolean
  id: string
}

export interface ChatRequest {
  input: string
  session_id: string
}

export interface StreamResponse {
  content?: string
  done?: boolean
}

export interface Session {
  id: string
  name: string
  createdAt: Date
  lastUsed: Date
}

// 新增：组件间通信的事件类型
export interface ChatEvents {
  sendMessage: (message: string) => void
  clearMessages: () => void
  changeSession: (sessionId: string) => void
  generateNewSession: () => void
  deleteSession: (sessionId: string) => void
}