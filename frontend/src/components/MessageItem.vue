<template>
  <!-- 只有当消息有内容时才渲染 -->
  <div v-if="hasContent" :class="['message', message.type]">
    <!-- AI消息：头像在左，消息在右 -->
    <template v-if="message.type === 'ai'">
      <div class="message-avatar">
        <div class="avatar ai">🤖</div>
      </div>
      <div class="message-content">
        <div class="message-header">
          <span class="message-sender">AI助手</span>
        </div>
        <div class="message-bubble">
          <div 
            class="message-text" 
            :class="{ streaming: message.isStreaming }"
            v-html="formattedText"
          ></div>
        </div>
        <div class="message-footer">
          <span class="message-time">{{ formatTime(message.timestamp) }}</span>
        </div>
      </div>
    </template>

    <!-- 用户消息：消息在左，头像在右 -->
    <template v-else>
      <div class="message-content">
        <div class="message-header">
          <span class="message-sender">你</span>
        </div>
        <div class="message-bubble">
          <div class="message-text" v-html="formattedText"></div>
        </div>
        <div class="message-footer">
          <span class="message-time">{{ formatTime(message.timestamp) }}</span>
        </div>
      </div>
      <div class="message-avatar">
        <div class="avatar user">👤</div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Message } from '@/types/chat'
import { renderMarkdown } from '@/utils/markdown'

const props = defineProps({
  message: {
    type: Object as () => Message,
    required: true
  }
})

// 是否有内容可显示
const hasContent = computed(() => {
  return props.message.text && props.message.text.trim().length > 0
})

const formattedText = computed(() => {
  if (!props.message.text) return ''
  
  if (props.message.isStreaming) {
    return props.message.text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\n/g, '<br>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
  }
  
  return renderMarkdown(props.message.text)
})

const formatTime = (timestamp: Date): string => {
  return timestamp.toLocaleTimeString('zh-CN', { 
    hour: '2-digit', 
    minute: '2-digit' 
  })
}
</script>

<style scoped>
.message {
  display: flex;
  gap: 1rem;
  margin-bottom: 1.5rem;
  align-items: flex-start;
}

.message.ai {
  justify-content: flex-start;
}

.message.user {
  justify-content: flex-end;
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

.avatar.user {
  background: linear-gradient(135deg, #3b82f6, #1d4ed8);
  border-color: #3b82f6;
}

.avatar.ai {
  background: linear-gradient(135deg, #8b5cf6, #7c3aed);
  border-color: #8b5cf6;
}

.message-content {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.message.ai .message-content {
  align-items: flex-start;
}

.message.user .message-content {
  align-items: flex-end;
}

.message-header {
  margin-bottom: 0.5rem;
  padding: 0 0.25rem;
}

.message-sender {
  font-weight: 600;
  font-size: 0.875rem;
  white-space: nowrap;
}

.message.user .message-sender {
  color: #3b82f6;
}

.message.ai .message-sender {
  color: #8b5cf6;
}

/* 气泡容器 - 关键：真正的自适应宽度 */
.message-bubble {
  display: inline-block;
  max-width: 100%;
}

/* 消息文本 - 核心样式 */
.message-text {
  background: #1a1a1a;
  border: 1px solid #333;
  border-radius: 20px;
  padding: 12px 16px;
  line-height: 1.6;
  word-wrap: break-word;
  color: #e5e5e5;
  /* 关键：让文本气泡根据内容自然调整宽度 */
  display: inline-block;
  max-width: 500px;
  min-width: 0;
}

/* AI消息气泡样式 */
.message.ai .message-text {
  border-bottom-left-radius: 4px;
  background: #1e1a2e;
  border-color: #8b5cf6;
}

/* 用户消息气泡样式 */
.message.user .message-text {
  border-bottom-right-radius: 4px;
  background: #1a2332;
  border-color: #3b82f6;
}

/* 时间显示 */
.message-footer {
  margin-top: 0.25rem;
  padding: 0 0.25rem;
}

.message.ai .message-footer {
  align-self: flex-start;
}

.message.user .message-footer {
  align-self: flex-end;
}

.message-time {
  font-size: 0.75rem;
  color: #888;
  opacity: 0.7;
  white-space: nowrap;
}

/* 流式输出效果 */
.message-text.streaming {
  border-color: #8b5cf6;
  box-shadow: 0 0 10px rgba(139, 92, 246, 0.2);
}

.message-text.streaming::after {
  content: '▋';
  color: #8b5cf6;
  animation: blink 1s infinite;
  margin-left: 2px;
}

@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* Markdown 样式 */
.message-text :deep(h1),
.message-text :deep(h2),
.message-text :deep(h3),
.message-text :deep(h4),
.message-text :deep(h5),
.message-text :deep(h6) {
  margin: 1rem 0 0.5rem 0;
  color: #e5e5e5;
}

.message-text :deep(h1:first-child),
.message-text :deep(h2:first-child),
.message-text :deep(h3:first-child),
.message-text :deep(h4:first-child),
.message-text :deep(h5:first-child),
.message-text :deep(h6:first-child) {
  margin-top: 0;
}

.message-text :deep(p) {
  margin: 0.5rem 0;
}

.message-text :deep(p:first-child) {
  margin-top: 0;
}

.message-text :deep(p:last-child) {
  margin-bottom: 0;
}

.message-text :deep(code) {
  background: #0a0a0a;
  color: #e5e5e5;
  padding: 0.2rem 0.4rem;
  border-radius: 0.25rem;
  font-family: 'SF Mono', Consolas, monospace;
  font-size: 0.875em;
}

.message-text :deep(pre) {
  background: #0a0a0a;
  border: 1px solid #404040;
  border-radius: 0.5rem;
  padding: 1rem;
  margin: 1rem 0;
  overflow-x: auto;
}

.message-text :deep(pre:first-child) {
  margin-top: 0;
}

.message-text :deep(pre:last-child) {
  margin-bottom: 0;
}

.message-text :deep(pre code) {
  background: none;
  padding: 0;
  color: inherit;
}

.message-text :deep(blockquote) {
  border-left: 3px solid #8b5cf6;
  padding-left: 1rem;
  margin: 1rem 0;
  color: #ccc;
}

.message-text :deep(blockquote:first-child) {
  margin-top: 0;
}

.message-text :deep(blockquote:last-child) {
  margin-bottom: 0;
}

.message-text :deep(ul),
.message-text :deep(ol) {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}

.message-text :deep(ul:first-child),
.message-text :deep(ol:first-child) {
  margin-top: 0;
}

.message-text :deep(ul:last-child),
.message-text :deep(ol:last-child) {
  margin-bottom: 0;
}

.message-text :deep(li) {
  margin: 0.25rem 0;
}

.message-text :deep(table) {
  border-collapse: collapse;
  margin: 1rem 0;
  width: 100%;
}

.message-text :deep(table:first-child) {
  margin-top: 0;
}

.message-text :deep(table:last-child) {
  margin-bottom: 0;
}

.message-text :deep(th),
.message-text :deep(td) {
  border: 1px solid #404040;
  padding: 0.5rem;
  text-align: left;
}

.message-text :deep(th) {
  background: #2d2d2d;
  font-weight: 600;
}

.message-text :deep(a) {
  color: #3b82f6;
  text-decoration: none;
}

.message-text :deep(a:hover) {
  text-decoration: underline;
}

.message-text :deep(strong) {
  color: #e5e5e5;
  font-weight: 600;
}

.message-text :deep(em) {
  color: #ccc;
  font-style: italic;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .message {
    gap: 0.75rem;
  }
  
  .avatar {
    width: 32px;
    height: 32px;
    font-size: 1rem;
  }
  
  .message-text {
    padding: 10px 14px;
    max-width: calc(100vw - 120px);
  }
}

@media (max-width: 480px) {
  .message-header {
    margin-bottom: 0.25rem;
  }
  
  .message-sender,
  .message-time {
    font-size: 0.8rem;
  }
  
  .message-footer {
    margin-top: 0.125rem;
  }
  
  .message-text {
    max-width: calc(100vw - 100px);
  }
}
</style>