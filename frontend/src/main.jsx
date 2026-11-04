import React, {memo, useEffect, useRef, useState} from 'react'
import {createRoot} from 'react-dom/client'
import {Alert, Collapse, ConfigProvider, Spin} from 'antd'
import {Bubble, Sender} from '@ant-design/x'
import {XStream} from '@ant-design/x-sdk'
import {XMarkdown} from '@ant-design/x-markdown'
import {ArrowUpOutlined, BookOutlined, LinkOutlined, ThunderboltFilled} from '@ant-design/icons'
import ReactMarkdown from 'react-markdown'
import '@ant-design/x-markdown/themes/light.css'
import '@ant-design/x-markdown/themes/dark.css'
import 'antd/dist/reset.css'
import './style.css'

async function request(path, {body, method = 'GET'} = {}) {
  const response = await fetch(`/api${path}`, {
    method,
    headers: body ? {'Content-Type': 'application/json'} : {},
    ...(body ? {body: JSON.stringify(body)} : {}),
  })
  const data = await response.json()
  if (!response.ok) {
    const error = new Error(typeof data.detail === 'string' ? data.detail : '请求失败，请稍后重试。')
    throw error
  }
  return data
}

function isArticleUrl(text) {
  try {
    const url = new URL(text)
    return !/\s/.test(text) && ['http:', 'https:'].includes(url.protocol)
  } catch {
    return false
  }
}

const ChatMessage = memo(function ChatMessage({item}) {
  return <div className={`message message-${item.role}`}>
    <div className="message-label">{item.role === 'user' ? '你' : 'BLOGMIND'}</div>
    {item.role === 'error' ? <Alert type="error" showIcon title={item.content}/> :
      <Bubble placement={item.role === 'user' ? 'end' : 'start'}
          content={item.role === 'assistant' ?
            <div className="markdown"><ReactMarkdown>{item.content}</ReactMarkdown></div>
            : item.content}/>}
    {item.role === 'assistant' && item.documents && (item.documents.length ?
      <Collapse className="sources" ghost items={[{
        key: 'sources', label: `查看检索资料（${item.documents.length}）`,
        children: item.documents.map((doc, i) => <div className="source" key={i}>
          <strong>[{i + 1}] {doc.title || '检索资料'}</strong>
          <div>{/^https?:\/\//i.test(doc.source) ?
            <a href={doc.source} target="_blank" rel="noreferrer">{doc.source}</a>
            : doc.source}</div>
          <p>{doc.content}</p>
        </div>),
      }]}/> : <p className="no-source">此回答未附带博客证据。</p>)}
  </div>
})

function Workspace() {
  const [busy, setBusy] = useState('')
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])
  const chatWindow = useRef(null)
  useEffect(() => {
    const element = chatWindow.current
    element?.scrollTo({top: element.scrollHeight, behavior: 'smooth'})
  }, [messages, busy])

  async function readStream(text) {

    const id = `answer-${Date.now()}`

    setMessages((previous) => [
      ...previous, {id, role: 'assistant', content: '', status: 'loading'},
    ])

    const response = await fetch('/api/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: text}),
    })
    if (!response.ok) {
      throw new Error('当前环境不支持流式响应。')
    }


    try {
      for await (const chunk of XStream({
        readableStream: response.body,

      })) {

        const payload = JSON.parse(chunk.data)

        setMessages((previous) => previous.map((item) => item.id === id ? {
          ...item,
          ...(payload.content ? {
            content: item.content + payload.content,
            status: 'updating',
          } : {}),
        } : item))
      }

    } catch (e) {
      setMessages((previous) => previous.map(
        (item) => item.id === id ? {...item, status: 'success'} : item,
      ))

    }
  }


  async function submit(value) {
    const text = value.trim()
    if (!text || busy) return
    const isUrl = isArticleUrl(text)
    setQuestion('')
    setMessages((previous) => [
      ...previous, {role: 'user', status: 'success', content: text, id: Date.now()},
    ])
    setBusy(isUrl ? 'import' : 'ask')
    try {
      if (isUrl) {
        const data = await request('/import', {method: 'POST', body: {url: text}})
        setMessages((previous) => [...previous, {
          role: 'assistant',
          content: `文章已导入，共 ${data.count} 个片段。现在可以向我提问了。`
        }])
      } else {

        await readStream(text)
      }
    } catch (error) {
      setMessages((previous) => [...previous, {
        role: 'error',
        content: `${isUrl ? '文章导入' : '回答生成'}失败：${error.message}`
      }])
    } finally {
      setBusy('')
    }
  }

  console.log(messages)
  return <div className="app-shell">
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark"><BookOutlined/></div>
        <div><strong>BlogMind</strong><span>AI 阅读助手</span></div>
      </div>
    </header>
    <main className="workspace">
      <div className="chat-heading">
        <div>
          <div className="eyebrow"><span/> YOUR READING SPACE</div>
          <h1>和你的文章，<span>聊聊。</span></h1><p>发送博客链接，或直接提出你想了解的问题。</p></div>
        <div className="heading-icon" aria-hidden="true"><ThunderboltFilled/></div>
      </div>
      <section className="chat-card" aria-label="博客问答">
        <div className="card-header">
          <div><span className="card-icon"><ThunderboltFilled/></span><strong>博客问答</strong><span
            className="header-caption">你的 AI 阅读伙伴</span></div>
          <span className="header-tag">AI ASSISTANT</span></div>
        <div className="chat-window" ref={chatWindow} aria-live="polite">
          {messages.length === 0 ?
            <div className="empty-chat">
              <div className="empty-icon"><BookOutlined/></div>
              <h2>从一篇文章开始</h2><p>粘贴文章链接导入知识库，或者直接向我提问。</p>
              <div className="suggestions">
                <button onClick={() => setQuestion('总结文章的核心观点')}>
                  <ThunderboltFilled/> 总结文章的核心观点 <ArrowUpOutlined/></button>
                <button onClick={() => setQuestion('文章有哪些关键细节？')}>
                  <LinkOutlined/> 文章有哪些关键细节？ <ArrowUpOutlined/></button>
              </div>
            </div> :
            < >
              <Bubble.List
                autoScroll
                role={{
                  assistant: {
                    placement: 'start',
                    header: (_, info) => <div className="message-label">BLOGMIND</div>,
                    contentRender: (content, {status}) => {
                      const newContent = content.replace(/\n\n/g, '<br/><br/>')
                      return (
                        <XMarkdown className="x-markdown-light" content={newContent}/>
                      )
                    },
                  },
                  user: {
                    placement: 'end',
                    header: (_, info) => <div
                      className="message-label">你</div>,
                  }
                }}

                items={messages.map(item => ({
                  ...item,
                  loading: item.status === 'loading',
                  typing: {
                    effect: 'typing',
                    step: 5,
                    interval: 50,
                  },
                  key: item.id,
                }))}
              />
            </>
          }
        </div>
        <div className="sender-wrap">
          <label>发送消息</label>
          <Sender aria-label="发送消息"
              autoSize={{minRows: 1, maxRows: 4}}
              value={question}
              onChange={setQuestion}
              onSubmit={submit}
              loading={!!busy}
              placeholder="输入问题，或粘贴博客文章链接…"/>
          <div className="sender-hint">
            <span><LinkOutlined/> 发送完整文章链接即可导入</span><span>回答由 AI 生成，请核对重要信息</span>
          </div>
        </div>
      </section>
      <footer>AI BLOG SEARCH <span>·</span> 阅读、提问、发现</footer>
    </main>
  </div>
}

createRoot(document.getElementById('root')).render(<ConfigProvider theme={{
  token: {
    colorPrimary: '#5264dc', borderRadius: 10,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", '
      + '"PingFang SC", "Microsoft YaHei", sans-serif',
  }
}}><Workspace/></ConfigProvider>)
