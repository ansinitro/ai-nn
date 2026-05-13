import { useState, useRef, useEffect } from 'react'

const EXAMPLES = [
  'How does transformer self-attention work?',
  'What are GANs and how do they train?',
  'Explain knowledge distillation',
  'How does federated learning preserve privacy?',
  'What is few-shot meta-learning?',
]

function useStatus() {
  const [status, setStatus] = useState({ status: 'loading', chunks: 0, llm: '' })
  useEffect(() => {
    const poll = async () => {
      try {
        const r = await fetch('/api/status')
        const d = await r.json()
        setStatus(d)
        if (d.status !== 'ready') setTimeout(poll, 3000)
      } catch { setTimeout(poll, 3000) }
    }
    poll()
  }, [])
  return status
}

function StatusPill({ status }) {
  const ready = status.status === 'ready'
  return (
    <div className={`status-pill ${ready ? 'ready' : ''}`}>
      <span className="dot" />
      {ready
        ? `Ready · ${status.chunks.toLocaleString()} chunks · ${status.llm}`
        : 'Pipeline loading…'}
    </div>
  )
}

function UserBubble({ text }) {
  return <div className="user-bubble">{text}</div>
}

function Thinking() {
  return (
    <div className="thinking">
      <span>Thinking</span>
      <span className="dots">
        <span /><span /><span />
      </span>
    </div>
  )
}

function AnswerCard({ result }) {
  return (
    <div className="answer-card">
      <div className="answer-header">🤖 Answer</div>
      <div className="answer-body">{result.answer}</div>
      <div className="metrics-row">
        <span className="metric">⏱ Retrieval: <strong>{result.retrieval_time_ms.toFixed(0)} ms</strong></span>
        <span className="metric">⚡ Generation: <strong>{result.generation_time_s.toFixed(1)} s</strong></span>
        <span className="metric">📄 Chunks used: <strong>{result.num_chunks}</strong></span>
      </div>
      {result.sources.length > 0 && (
        <div className="sources-section">
          <div className="sources-title">📚 Sources Retrieved</div>
          <div className="source-cards">
            {result.sources.map((s, i) => (
              <div className="source-card" key={i}>
                <div className="source-rank">{i + 1}</div>
                <div className="source-info">
                  <div className="source-name">{s.doc_name}</div>
                  <div className="source-preview">{s.text_preview}</div>
                </div>
                <div className="source-score">{(s.score * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default function App() {
  const status = useStatus()
  const [messages, setMessages] = useState([]) // [{q, result, loading}]
  const [input, setInput] = useState('')
  const [topK, setTopK] = useState(5)
  const [busy, setBusy] = useState(false)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const submit = async (question) => {
    const q = (question || input).trim()
    if (!q || busy || status.status !== 'ready') return
    setInput('')
    setBusy(true)
    const idx = messages.length
    setMessages(prev => [...prev, { q, result: null, loading: true }])

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, top_k: topK }),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      setMessages(prev => prev.map((m, i) => i === idx ? { ...m, result, loading: false } : m))
    } catch (e) {
      setMessages(prev => prev.map((m, i) => i === idx
        ? { ...m, result: { answer: `Error: ${e.message}`, sources: [], retrieval_time_ms: 0, generation_time_s: 0, num_chunks: 0 }, loading: false }
        : m))
    } finally {
      setBusy(false)
    }
  }

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-left">
          <div className="logo">🔬</div>
          <div>
            <h1>RAG QA System</h1>
            <p>ML Research Paper Q&A · Variant 5 · AAI-2501</p>
          </div>
        </div>
        <StatusPill status={status} />
      </header>

      {/* Chat */}
      <main className="chat-area">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="icon">📖</div>
            <h2>Ask anything about ML research</h2>
            <p>Powered by Llama 2 7B + SBERT + FAISS over 85 arXiv papers</p>
            <div className="examples">
              {EXAMPLES.map(e => (
                <button key={e} className="example-chip" onClick={() => submit(e)}>{e}</button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className="message">
              <UserBubble text={m.q} />
              {m.loading ? <Thinking /> : <AnswerCard result={m.result} />}
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </main>

      {/* Input bar */}
      <div className="input-bar">
        <div className="topk-row">
          <span>Sources: <strong>{topK}</strong></span>
          <input type="range" min={1} max={10} value={topK} onChange={e => setTopK(+e.target.value)} />
        </div>
        <div className="input-wrap">
          <textarea
            ref={textareaRef}
            rows={1}
            placeholder="Ask a question about ML papers… (Enter to send)"
            value={input}
            onChange={e => { setInput(e.target.value); e.target.style.height = 'auto'; e.target.style.height = e.target.scrollHeight + 'px' }}
            onKeyDown={onKey}
            disabled={busy || status.status !== 'ready'}
          />
          <button className="send-btn" onClick={() => submit()} disabled={busy || !input.trim() || status.status !== 'ready'}>
            {busy ? '⏳' : '➤'}
          </button>
        </div>
        <div className="input-hint">Press Enter to send · Shift+Enter for new line</div>
      </div>
    </div>
  )
}
