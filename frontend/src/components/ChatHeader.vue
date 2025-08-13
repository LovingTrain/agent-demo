<template>
  <div class="chat-header">
    <div class="header-left">
      <button 
        v-if="sidebarCollapsed || isMobile"
        @click="toggleSidebar" 
        class="sidebar-toggle"
        title="打开侧边栏"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M3 12h18M3 6h18M3 18h18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        </svg>
      </button>
      <h2>AI 聊天助手</h2>
    </div>
    <div class="header-right">
      <span class="current-session">{{ currentSessionId }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  currentSessionId: string
  sidebarCollapsed: boolean
  isMobile: boolean
}

interface Emits {
  (e: 'toggle-sidebar'): void
}

defineProps<Props>()
const emit = defineEmits<Emits>()

const toggleSidebar = () => {
  emit('toggle-sidebar')
}
</script>

<style scoped>
.chat-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
  border-bottom: 1px solid #333;
  padding: 1rem 1.5rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.sidebar-toggle {
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 0.375rem;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.sidebar-toggle:hover {
  color: #e5e5e5;
  background: #2d2d2d;
}

.header-left h2 {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.current-session {
  font-family: 'SF Mono', Consolas, monospace;
  font-size: 0.875rem;
  color: #888;
  background: #2d2d2d;
  padding: 0.25rem 0.75rem;
  border-radius: 1rem;
}

@media (max-width: 768px) {
  .chat-header {
    padding: 0.75rem 1rem;
  }
  
  .current-session {
    display: none;
  }
}
</style>