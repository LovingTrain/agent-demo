<template>
  <div class="welcome-message">
    <div class="welcome-content">
      <h3>👋 欢迎使用AI聊天助手</h3>
      <p>开始对话吧！我支持 Markdown 格式的回复。</p>
      <div class="welcome-tips">
        <div class="tip">💡 支持代码高亮</div>
        <div class="tip">📝 支持多行文本</div>
        <div class="tip">🔄 流式输出</div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// 无需逻辑
</script>

<style scoped>
.welcome-message {
  display: flex;
  justify-content: center;
  align-items: center;
  flex: 1;
  padding: 2rem;
}

.welcome-content {
  text-align: center;
  max-width: 400px;
}

.welcome-content h3 {
  margin-bottom: 1rem;
  color: #e5e5e5;
  font-size: 1.5rem;
}

.welcome-content p {
  color: #ccc;
  margin-bottom: 1.5rem;
  line-height: 1.6;
}

.welcome-tips {
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}

.tip {
  background: #1a1a1a;
  border: 1px solid #333;
  padding: 0.5rem 1rem;
  border-radius: 1rem;
  font-size: 0.875rem;
  color: #ccc;
}

@media (max-width: 768px) {
  .welcome-tips {
    flex-direction: column;
    align-items: center;
  }
}
</style>