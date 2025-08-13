<template>
  <div class="chat-input">
    <div class="input-container">
      <div class="input-wrapper">
        <textarea
          v-model="inputMessage"
          @keydown="handleKeyDown"
          @input="adjustTextareaHeight"
          placeholder="输入你的消息... (支持 Markdown 格式)"
          ref="textareaRef"
          :disabled="isLoading"
          class="message-input"
        ></textarea>
        <div class="input-actions">
          <button 
            @click="clearMessages" 
            class="clear-btn"
            title="清空对话"
            :disabled="isLoading"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M3 6h18M8 6V4a2 2 0 012-2h4a2 2 0 012 2v2m3 0v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6h14z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
          <button 
            @click="sendMessage" 
            :disabled="isLoading || !inputMessage.trim()"
            class="send-btn"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M2 21L23 12L2 3V10L17 12L2 14V21Z" fill="currentColor"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
    <div class="input-footer">
      <div class="shortcuts">
        <kbd>Enter</kbd> 发送 • <kbd>Shift + Enter</kbd> 换行
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, type Ref } from 'vue'

interface Props {
  isLoading: boolean
}

interface Emits {
  (e: 'send-message', message: string): void
  (e: 'clear-messages'): void
}

defineProps<Props>()
const emit = defineEmits<Emits>()

const inputMessage = ref<string>('')
const textareaRef: Ref<HTMLTextAreaElement | null> = ref(null)

const adjustTextareaHeight = (): void => {
  nextTick(() => {
    if (textareaRef.value) {
      textareaRef.value.style.height = 'auto'
      textareaRef.value.style.height = Math.min(textareaRef.value.scrollHeight, 120) + 'px'
    }
  })
}

const sendMessage = (): void => {
  if (!inputMessage.value.trim()) return
  
  const message = inputMessage.value.trim()
  inputMessage.value = ''
  adjustTextareaHeight()
  
  emit('send-message', message)
}

const clearMessages = (): void => {
  if (confirm('确定要清空当前对话吗？')) {
    emit('clear-messages')
  }
}

const handleKeyDown = (event: KeyboardEvent): void => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendMessage()
  }
}
</script>

<style scoped>
.chat-input {
  background: #1a1a1a;
  border-top: 1px solid #333;
  padding: 1rem 1.5rem;
}

.input-container {
  margin-bottom: 0.75rem;
}

.input-wrapper {
  display: flex;
  background: #0a0a0a;
  border: 1px solid #404040;
  border-radius: 1rem;
  overflow: hidden;
  transition: border-color 0.2s;
}

.input-wrapper:focus-within {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.message-input {
  flex: 1;
  background: none;
  border: none;
  color: #e5e5e5;
  padding: 1rem 1.25rem;
  resize: none;
  outline: none;
  font-family: inherit;
  font-size: 1rem;
  line-height: 1.5;
  min-height: 20px;
  max-height: 120px;
}

.message-input::placeholder {
  color: #666;
}

.message-input:disabled {
  color: #666;
  cursor: not-allowed;
}

.input-actions {
  display: flex;
  align-items: flex-end;
  padding: 0.5rem;
  gap: 0.5rem;
}

.clear-btn,
.send-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: all 0.2s;
  width: 40px;
  height: 40px;
}

.clear-btn {
  background: #2d2d2d;
  color: #888;
}

.clear-btn:hover:not(:disabled) {
  background: #dc2626;
  color: white;
}

.send-btn {
  background: #3b82f6;
  color: white;
}

.send-btn:hover:not(:disabled) {
  background: #2563eb;
  transform: translateY(-1px);
}

.clear-btn:disabled,
.send-btn:disabled {
  background: #404040;
  color: #666;
  cursor: not-allowed;
  transform: none;
}

.input-footer {
  display: flex;
  justify-content: center;
  align-items: center;
  font-size: 0.75rem;
  color: #888;
}

.shortcuts {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.shortcuts kbd {
  background: #2d2d2d;
  border: 1px solid #404040;
  border-radius: 0.25rem;
  padding: 0.1rem 0.3rem;
  font-family: inherit;
  font-size: 0.7rem;
}

@media (max-width: 768px) {
  .chat-input {
    padding: 0.75rem 1rem;
  }
}
</style>