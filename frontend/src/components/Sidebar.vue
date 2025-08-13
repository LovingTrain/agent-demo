<template>
  <div :class="['sidebar', { collapsed: isCollapsed, 'mobile-overlay': isMobileOverlay }]">
    <!-- 侧边栏头部 -->
    <div class="sidebar-header">
      <div class="sidebar-title" v-if="!isCollapsed">
        <h3>会话历史</h3>
        <button @click="toggleSidebar" class="collapse-btn" title="收起侧边栏">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M15 18l-6-6 6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
      <button v-else @click="toggleSidebar" class="expand-btn" title="展开侧边栏">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M9 18l6-6-6-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>
    </div>

    <!-- 侧边栏内容 -->
    <div class="sidebar-content" v-if="!isCollapsed">
      <!-- 搜索框 -->
      <div class="search-section">
        <div class="search-input-wrapper">
          <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none">
            <circle cx="11" cy="11" r="8" stroke="currentColor" stroke-width="2"/>
            <path d="m21 21-4.35-4.35" stroke="currentColor" stroke-width="2"/>
          </svg>
          <input
            v-model="searchQuery"
            @input="filterSessions"
            placeholder="搜索会话..."
            class="search-input"
          />
          <button 
            v-if="searchQuery" 
            @click="clearSearch"
            class="clear-search-btn"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2"/>
            </svg>
          </button>
        </div>
      </div>

      <!-- 新建会话 -->
      <div class="new-session-section">
        <button @click="startNewSession" class="new-session-btn">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M12 5v14m7-7H5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
          <span>新建会话</span>
        </button>
      </div>

      <!-- 新建会话输入框（只在创建时显示） -->
      <div v-if="showNewSessionInput" class="create-session-section">
        <div class="create-input-wrapper">
          <input
            v-model="newSessionId"
            @keydown="handleNewSessionKeyDown"
            @blur="cancelNewSession"
            @input="validateNewSessionInput"
            placeholder="输入会话名称"
            class="create-input"
            ref="newSessionInputRef"
            maxlength="50"
          />
          <button 
            @click="confirmNewSession"
            :disabled="!newSessionId.trim() || !!inputError"
            class="confirm-btn"
            title="确认创建"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M20 6L9 17L4 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </button>
          <button @click="cancelNewSession" class="cancel-btn" title="取消">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
              <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" stroke-width="2"/>
            </svg>
          </button>
        </div>
        <div v-if="inputError" class="input-error">{{ inputError }}</div>
      </div>

      <!-- 会话列表 -->
      <div class="sessions-section">
        <div v-if="filteredSessions.length === 0" class="no-sessions">
          {{ searchQuery ? '未找到匹配的会话' : '暂无历史会话' }}
        </div>
        <div v-else class="sessions-list">
          <div
            v-for="session in filteredSessions"
            :key="session.id"
            @click="selectSession(session.id)"
            @contextmenu.prevent="showContextMenu($event, session)"
            :class="['session-item', { active: session.id === currentSessionId }]"
          >
            <div class="session-info">
              <div class="session-name">{{ session.id }}</div>
              <div class="session-time">{{ formatSessionTime(session.lastUsed) }}</div>
            </div>
            <div class="session-actions">
              <button 
                @click.stop="showContextMenu($event, session)"
                class="more-btn"
                title="更多操作"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="1" stroke="currentColor" stroke-width="2"/>
                  <circle cx="19" cy="12" r="1" stroke="currentColor" stroke-width="2"/>
                  <circle cx="5" cy="12" r="1" stroke="currentColor" stroke-width="2"/>
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 右键菜单 -->
    <div
      v-if="contextMenu.show"
      :style="{ top: contextMenu.y + 'px', left: contextMenu.x + 'px' }"
      class="context-menu"
      @click.stop
    >
      <button @click="renameSession" class="context-menu-item">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="M12 20h9M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" stroke="currentColor" stroke-width="2"/>
        </svg>
        重命名
      </button>
      <button @click="duplicateSession" class="context-menu-item">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2" stroke="currentColor" stroke-width="2"/>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" stroke="currentColor" stroke-width="2"/>
        </svg>
        复制会话
      </button>
      <div class="context-menu-divider"></div>
      <button @click="deleteSessionFromMenu" class="context-menu-item danger">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6h14z" stroke="currentColor" stroke-width="2"/>
        </svg>
        删除会话
      </button>
    </div>

    <!-- 移动端遮罩 -->
    <div v-if="isMobileOverlay && !isCollapsed" @click="closeSidebar" class="mobile-backdrop"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, onMounted, onUnmounted } from 'vue'
import type { Session } from '@/types/chat'

interface Props {
  currentSessionId: string
  sessions: Session[]
  isCollapsed: boolean
  isMobile: boolean
}

interface Emits {
  (e: 'change-session', sessionId: string): void
  (e: 'create-new-session', sessionId: string): void
  (e: 'delete-session', sessionId: string): void
  (e: 'rename-session', sessionId: string, newName: string): void
  (e: 'duplicate-session', sessionId: string): void
  (e: 'toggle-sidebar'): void
  (e: 'close-sidebar'): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const searchQuery = ref('')
const showNewSessionInput = ref(false)
const newSessionId = ref('')
const inputError = ref('')
const newSessionInputRef = ref<HTMLInputElement>()

const contextMenu = ref({
  show: false,
  x: 0,
  y: 0,
  session: null as Session | null
})

const filteredSessions = ref<Session[]>([])

const isMobileOverlay = computed(() => props.isMobile && !props.isCollapsed)

// 过滤会话
const filterSessions = () => {
  if (!searchQuery.value.trim()) {
    filteredSessions.value = [...props.sessions]
  } else {
    const query = searchQuery.value.toLowerCase()
    filteredSessions.value = props.sessions.filter(session =>
      session.id.toLowerCase().includes(query)
    )
  }
}

const clearSearch = () => {
  searchQuery.value = ''
  filterSessions()
}

const toggleSidebar = () => {
  emit('toggle-sidebar')
}

const closeSidebar = () => {
  emit('close-sidebar')
}

const selectSession = (sessionId: string) => {
  emit('change-session', sessionId)
  if (props.isMobile) {
    closeSidebar()
  }
}

const startNewSession = () => {
  showNewSessionInput.value = true
  nextTick(() => {
    newSessionInputRef.value?.focus()
  })
}

const cancelNewSession = () => {
  showNewSessionInput.value = false
  newSessionId.value = ''
  inputError.value = ''
}

const validateNewSessionInput = () => {
  const value = newSessionId.value.trim()
  inputError.value = ''
  
  if (value.length === 0) return
  
  if (props.sessions.some(s => s.id === value)) {
    inputError.value = '会话名称已存在'
    return
  }
  
  if (!/^[a-zA-Z0-9_\-\u4e00-\u9fa5\s]+$/.test(value)) {
    inputError.value = '包含无效字符'
    return
  }
}

const confirmNewSession = () => {
  const sessionId = newSessionId.value.trim()
  if (!sessionId || inputError.value) return
  
  emit('create-new-session', sessionId)
  cancelNewSession()
}

const handleNewSessionKeyDown = (event: KeyboardEvent) => {
  if (event.key === 'Enter') {
    event.preventDefault()
    confirmNewSession()
  } else if (event.key === 'Escape') {
    cancelNewSession()
  }
}

const showContextMenu = (event: MouseEvent, session: Session) => {
  contextMenu.value = {
    show: true,
    x: event.clientX,
    y: event.clientY,
    session
  }
}

const hideContextMenu = () => {
  contextMenu.value.show = false
}

const renameSession = () => {
  if (!contextMenu.value.session) return
  const newName = prompt('请输入新的会话名称:', contextMenu.value.session.id)
  if (newName && newName.trim() && newName !== contextMenu.value.session.id) {
    emit('rename-session', contextMenu.value.session.id, newName.trim())
  }
  hideContextMenu()
}

const duplicateSession = () => {
  if (!contextMenu.value.session) return
  emit('duplicate-session', contextMenu.value.session.id)
  hideContextMenu()
}

const deleteSessionFromMenu = () => {
  if (!contextMenu.value.session) return
  if (confirm(`确定要删除会话 "${contextMenu.value.session.id}" 吗？`)) {
    emit('delete-session', contextMenu.value.session.id)
  }
  hideContextMenu()
}

const formatSessionTime = (timestamp: Date): string => {
  const now = new Date()
  const diff = now.getTime() - timestamp.getTime()
  const hours = Math.floor(diff / (1000 * 60 * 60))
  
  if (hours < 1) return '刚刚'
  if (hours < 24) return `${hours}小时前`
  
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}天前`
  
  return timestamp.toLocaleDateString('zh-CN')
}

const handleClickOutside = (event: Event) => {
  if (contextMenu.value.show) {
    hideContextMenu()
  }
}

onMounted(() => {
  filterSessions()
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})

// 监听 sessions 变化
import { watch } from 'vue'
watch(() => props.sessions, () => {
  filterSessions()
}, { deep: true })
</script>

<style scoped>
.sidebar {
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #1a1a1a;
  border-right: 1px solid #333;
  transition: all 0.3s ease;
  width: 280px;
  position: relative;
  z-index: 100;
}

.sidebar.collapsed {
  width: 60px;
}

.sidebar.mobile-overlay {
  position: fixed;
  top: 0;
  left: 0;
  z-index: 1000;
  box-shadow: 2px 0 10px rgba(0, 0, 0, 0.3);
}

.mobile-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: -1;
}

/* 头部 */
.sidebar-header {
  padding: 1rem;
  border-bottom: 1px solid #333;
  flex-shrink: 0;
}

.sidebar-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.sidebar-title h3 {
  margin: 0;
  color: #e5e5e5;
  font-size: 1rem;
  font-weight: 600;
}

.collapse-btn,
.expand-btn {
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  padding: 0.25rem;
  border-radius: 0.25rem;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.collapse-btn:hover,
.expand-btn:hover {
  color: #e5e5e5;
  background: #2d2d2d;
}

.expand-btn {
  width: 100%;
  height: 40px;
}

/* 侧边栏内容 */
.sidebar-content {
  display: flex;
  flex-direction: column;
  flex: 1;
  overflow: hidden;
}

/* 搜索区域 */
.search-section {
  padding: 1rem;
  border-bottom: 1px solid #333;
}

.search-input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.search-icon {
  position: absolute;
  left: 0.75rem;
  color: #666;
  pointer-events: none;
}

.search-input {
  width: 100%;
  background: #0a0a0a;
  border: 1px solid #404040;
  color: #e5e5e5;
  padding: 0.5rem 0.5rem 0.5rem 2.5rem;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  outline: none;
  transition: border-color 0.2s;
}

.search-input:focus {
  border-color: #3b82f6;
}

.search-input::placeholder {
  color: #666;
}

.clear-search-btn {
  position: absolute;
  right: 0.5rem;
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  padding: 0.25rem;
  border-radius: 0.25rem;
  transition: color 0.2s;
}

.clear-search-btn:hover {
  color: #e5e5e5;
}

/* 新建会话区域 */
.new-session-section {
  padding: 1rem;
  border-bottom: 1px solid #333;
}

.new-session-btn {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: #3b82f6;
  border: none;
  color: white;
  padding: 0.75rem 1rem;
  border-radius: 0.5rem;
  cursor: pointer;
  font-size: 0.875rem;
  font-weight: 500;
  transition: all 0.2s;
}

.new-session-btn:hover {
  background: #2563eb;
}

/* 创建会话区域 */
.create-session-section {
  padding: 0 1rem 1rem 1rem;
  border-bottom: 1px solid #333;
}

.create-input-wrapper {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.create-input {
  flex: 1;
  background: #0a0a0a;
  border: 1px solid #404040;
  color: #e5e5e5;
  padding: 0.5rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  outline: none;
  transition: border-color 0.2s;
}

.create-input:focus {
  border-color: #3b82f6;
}

.create-input::placeholder {
  color: #666;
}

.confirm-btn,
.cancel-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  padding: 0.5rem;
  border-radius: 0.375rem;
  cursor: pointer;
  transition: all 0.2s;
  width: 32px;
  height: 32px;
  flex-shrink: 0;
}

.confirm-btn {
  background: #22c55e;
  color: white;
}

.confirm-btn:hover:not(:disabled) {
  background: #16a34a;
}

.confirm-btn:disabled {
  background: #404040;
  color: #666;
  cursor: not-allowed;
}

.cancel-btn {
  background: #ef4444;
  color: white;
}

.cancel-btn:hover {
  background: #dc2626;
}

.input-error {
  font-size: 0.75rem;
  color: #ef4444;
  margin-top: 0.25rem;
}

/* 会话列表区域 */
.sessions-section {
  flex: 1;
  overflow-y: auto;
  padding: 0.5rem 0;
}

.no-sessions {
  padding: 2rem 1rem;
  text-align: center;
  color: #888;
  font-size: 0.875rem;
}

.sessions-list {
  padding: 0 0.5rem;
}

.session-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0.75rem;
  margin-bottom: 0.25rem;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: all 0.2s;
  border: 1px solid transparent;
  position: relative;
}

.session-item:hover {
  background: #2d2d2d;
}

.session-item.active {
  background: #1e3a8a;
  border-color: #3b82f6;
}

.session-info {
  flex: 1;
  min-width: 0;
}

.session-name {
  font-size: 0.875rem;
  color: #e5e5e5;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 0.25rem;
}

.session-time {
  font-size: 0.75rem;
  color: #888;
}

.session-actions {
  opacity: 0;
  transition: opacity 0.2s;
}

.session-item:hover .session-actions {
  opacity: 1;
}

.more-btn {
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  padding: 0.25rem;
  border-radius: 0.25rem;
  transition: all 0.2s;
}

.more-btn:hover {
  color: #e5e5e5;
  background: #404040;
}

/* 右键菜单 */
.context-menu {
  position: fixed;
  background: #2d2d2d;
  border: 1px solid #404040;
  border-radius: 0.5rem;
  padding: 0.5rem 0;
  z-index: 1001;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
  min-width: 150px;
}

.context-menu-item {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: none;
  border: none;
  color: #e5e5e5;
  padding: 0.5rem 1rem;
  cursor: pointer;
  font-size: 0.875rem;
  transition: background 0.2s;
  text-align: left;
}

.context-menu-item:hover {
  background: #404040;
}

.context-menu-item.danger {
  color: #ef4444;
}

.context-menu-item.danger:hover {
  background: #ef4444;
  color: white;
}

.context-menu-divider {
  height: 1px;
  background: #404040;
  margin: 0.5rem 0;
}

/* 滚动条样式 */
.sessions-section::-webkit-scrollbar {
  width: 6px;
}

.sessions-section::-webkit-scrollbar-track {
  background: transparent;
}

.sessions-section::-webkit-scrollbar-thumb {
  background: #404040;
  border-radius: 3px;
}

.sessions-section::-webkit-scrollbar-thumb:hover {
  background: #555;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .sidebar {
    width: 300px;
  }
  
  .sidebar.collapsed {
    width: 0;
    overflow: hidden;
  }
}
</style>