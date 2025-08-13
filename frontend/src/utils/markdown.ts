import { marked } from 'marked'
import hljs from 'highlight.js'

// 配置 marked
marked.setOptions({
  highlight: function (code, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return hljs.highlight(code, { language: lang }).value
      } catch (err) {
        console.error('Highlight error:', err)
      }
    }
    return hljs.highlightAuto(code).value
  },
  breaks: true,
  gfm: true
})

export const renderMarkdown = (text: string): string => {
  try {
    return marked(text) as string
  } catch (error) {
    console.error('Markdown render error:', error)
    return text.replace(/\n/g, '<br>')
  }
}

export const isMarkdownComplete = (text: string): boolean => {
  // 简单检查markdown是否可能完整
  const codeBlockCount = (text.match(/```/g) || []).length
  const inlineCodeCount = (text.match(/`/g) || []).length - codeBlockCount * 3

  return codeBlockCount % 2 === 0 && inlineCodeCount % 2 === 0
}