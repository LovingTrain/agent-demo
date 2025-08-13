<template>
  <div class="chat-messages" ref="messagesContainer">
    <WelcomeMessage v-if="messages.length === 0" />
    
    <MessageItem
      v-for="message in messages"
      :key="message.id"
      :message="message"
    />
    
    <!-- 等待AI回复时的紧凑加载状态 -->
    <div v-if="isLoading && !currentStreamingMessage" class="loading-message">
      <div class="message-avatar">
        <div class="avatar ai">🤖</div>
      </div>
      <div class="loading-content">
        <div class="message-header">
          <span class="message-sender">AI助手</span>
        </div>
        <div class="compact-typing-indicator">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, type Ref } from 'vue'
import type { Message } from '@/types/chat'
import WelcomeMessage from './WelcomeMessage.vue'
import MessageItem from './MessageItem.vue'

interface Props {
  messages: Message[]
  isLoading: boolean
  currentStreamingMessage: Message | null
}

defineProps<Props>()

const messagesContainer: Ref<HTMLElement | null> = ref(null)

const scrollToBottom = (): void => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

// 暴露给父组件使用
defineExpose({
  scrollToBottom
})
</script>

<style scoped>
.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem 1.5rem;
  scroll-behavior: smooth;
}

/* 加载消息样式 */
.loading-message {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  align-items: flex-start;
}

.message-avatar {
  flex-shrink: 0;
}

.avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.25rem;
  border: 2px solid;
}

.avatar.ai {
  background: linear-gradient(135deg, #8b5cf6, #7c3aed);
  border-color: #8b5cf6;
}

.loading-content {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.message-header {
  margin-bottom: 0.5rem;
  padding: 0 0.25rem;
}

.message-sender {
  font-weight: 600;
  font-size: 0.875rem;
  color: #8b5cf6;
}

/* 紧凑的跳动点指示器 */
.compact-typing-indicator {
  display: inline-flex; /* 使用 inline-flex 让它紧凑 */
  gap: 0.25rem;
  align-items: center;
  padding: 0.75rem 1rem; /* 减小 padding */
  background: #1e1a2e; /* 使用AI消息的背景色 */
  border: 1px solid #8b5cf6;
  border-radius: 20px;
  border-bottom-left-radius: 4px; /* 添加小尖角 */
  width: fit-content; /* 关键：让宽度根据内容调整 */
  min-width: 60px; /* 最小宽度只够容纳三个点 */
  box-shadow: 0 0 10px rgba(139, 92, 246, 0.2);
}

.compact-typing-indicator span {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #8b5cf6;
  animation: bounce 1.4s infinite ease-in-out;
}

.compact-typing-indicator span:nth-child(1) {
  animation-delay: -0.32s;
}

.compact-typing-indicator span:nth-child(2) {
  animation-delay: -0.16s;
}

.compact-typing-indicator span:nth-child(3) {
  animation-delay: 0s;
}

@keyframes bounce {
  0%, 80%, 100% {
    transform: scale(0.8) translateY(0);
    opacity: 0.5;
  }
  40% {
    transform: scale(1) translateY(-4px);
    opacity: 1;
  }
}

.chat-messages::-webkit-scrollbar {
  width: 6px;
}

.chat-messages::-webkit-scrollbar-track {
  background: #1a1a1a;
}

.chat-messages::-webkit-scrollbar-thumb {
  background: #404040;
  border-radius: 3px;
}

.chat-messages::-webkit-scrollbar-thumb:hover {
  background: #555;
}

@media (max-width: 768px) {
  .chat-messages {
    padding: 1rem;
  }
  
  .loading-message {
    gap: 0.75rem;
  }
  
  .avatar {
    width: 32px;
    height: 32px;
    font-size: 1rem;
  }
  
  .compact-typing-indicator {
    padding: 0.5rem 0.75rem;
    min-width: 50px;
  }
  
  .compact-typing-indicator span {
    width: 5px;
    height: 5px;
  }
}
</style>