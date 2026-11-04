import React, {useEffect, useRef, useState} from 'react'
import {createRoot} from 'react-dom/client'
import {ConfigProvider} from 'antd'
import {Bubble, Sender, Sources} from '@ant-design/x'
import {readAnswerStream} from './answerStream'
import {XMarkdown} from '@ant-design/x-markdown'
import {ArrowUpOutlined, BookOutlined, LinkOutlined, ThunderboltFilled} from '@ant-design/icons'
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


function Workspace() {
    const [busy, setBusy] = useState('')
    const [question, setQuestion] = useState('')
    const [messages, setMessages] = useState([])
    const chatWindow = useRef(null)
    useEffect(() => {
        const element = chatWindow.current
        element?.scrollTo({top: element.scrollHeight, behavior: 'smooth'})
    }, [messages, busy])

    async function readStream(text, id) {

        try {
            const response = await fetch('/api/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: text}),
            })
            if (!response.ok || !response.body) {
                throw new Error('无法获取流式响应，请稍后重试。')
            }

            await readAnswerStream(response.body, (payload) => {
                setMessages((previous) => previous.map((item) => {
                    if (item.id !== id) return item
                    if (payload.type === 'content') {
                        return {
                            ...item,
                            content: item.content + payload.content,
                            status: 'updating',
                        }
                    }
                    if (payload.type === 'references') {
                        return {...item, documents: payload.documents}
                    }
                    return item
                }))
            })
            setMessages((previous) => previous.map(
                (item) => item.id === id ? {...item, status: 'success'} : item,
            ))
        } catch (error) {
            setMessages((previous) => previous.map(
                (item) => item.id === id ? {...item, status: 'error'} : item,
            ))
            throw error
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
            const id = `answer-${Date.now()}`

            setMessages((previous) => [
                ...previous, {id, role: 'assistant', content: '', documents: [], status: 'loading'},
            ])

            if (isUrl) {
                try {
                    const data = await request('/import', {method: 'POST', body: {url: text}})
                    setMessages((previous) => [...previous, {
                        role: 'assistant',
                        status: 'success',
                        content: `文章已导入，共 ${data.count} 个片段。现在可以向我提问了。`
                    }])
                } catch (error) {
                    setMessages((previous) => previous.map(
                        (item) => item.id === id ? {
                            ...item,
                            status: 'error',
                            content: `${isUrl ? '文章导入' : '回答生成'}失败：${error.message}`
                        } : item,
                    ))
                }

            } else {
                await readStream(text, id)
            }
        } finally {
            setBusy('')
        }
    }

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
                                            console.log(content)
                                            return (
                                                <XMarkdown
                                                    className="x-markdown-light"
                                                    content={content}
                                                    streaming={{
                                                        hasNextChunk: status === "updating",
                                                        enableAnimation: true,
                                                    }}
                                                />
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
                                    key: item.id,
                                    footer: item.role === 'assistant' && item.documents?.length ? (
                                        <Sources
                                            defaultExpanded={false}
                                            title={`参考资料（${item.documents.length}）`}
                                            items={item.documents.map((doc) => ({
                                                key: doc.index,
                                                title: `[${doc.index}] ${doc.title || doc.source || '参考资料'}`,
                                                url: /^https?:\/\//i.test(doc.source) ? doc.source : undefined,
                                            }))}
                                        />
                                    ) : null,
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
