import { useState, useRef, useEffect, useCallback } from 'react'

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

function Thinking() {
  return (
    <div className="thinking">
      <span>Generating answer</span>
      <span className="dots"><span /><span /><span /></span>
    </div>
  )
}

// Format raw chunk text into readable paragraphs
function formatText(text) {
  // Split on double newlines or sentence-ending patterns that look like paragraph breaks
  return text
    .replace(/\r\n/g, '\n')
    .split(/\n{2,}/)
    .map(p => p.replace(/\n/g, ' ').replace(/\s+/g, ' ').trim())
    .filter(p => p.length > 0)
}

// Modal for full chunk text
function ChunkModal({ source, onClose }) {
  const [copied, setCopied] = useState(false)
  const paragraphs = formatText(source.full_text)
  const wordCount = source.full_text.split(/\s+/).length

  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const handleCopy = () => {
    navigator.clipboard.writeText(source.full_text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div className="modal-header-left">
            <div className="modal-source-tag">📄 Source Document</div>
            <div className="modal-title">{source.doc_name}</div>
            <div className="modal-meta-row">
              <span className="modal-badge score">
                {(source.score * 100).toFixed(1)}% relevance
              </span>
              {source.used_in_context && (
                <span className="modal-badge used">✓ Used in prompt</span>
              )}
              <span className="modal-badge neutral">~{wordCount} words</span>
            </div>
          </div>
          <div className="modal-actions">
            <a
              className="modal-btn pdf-btn"
              href={`/papers/${source.doc_name}`}
              target="_blank"
              rel="noopener noreferrer"
              title="Open full PDF in new tab"
            >
              📄 Open PDF
            </a>
            <button className="modal-btn" onClick={handleCopy}>
              {copied ? '✓ Copied' : '⎘ Copy'}
            </button>
            <button className="modal-btn close" onClick={onClose}>✕</button>
          </div>
        </div>

        {/* Divider */}
        <div className="modal-divider" />

        {/* Body — formatted paragraphs */}
        <div className="modal-body">
          {paragraphs.map((para, i) => (
            <p key={i} className="modal-para">{para}</p>
          ))}
        </div>

        {/* Footer */}
        <div className="modal-footer">
          <span>Chunk #{source.chunk_id}</span>
          <span>Press <kbd>Esc</kbd> to close</span>
        </div>
      </div>
    </div>
  )
}


function SourceCard({ source, rank, onClick }) {
  return (
    <div className={`source-card ${source.used_in_context ? 'used' : ''}`} onClick={onClick} title="Click to read full passage">
      <div className="source-rank">{rank}</div>
      <div className="source-info">
        <div className="source-name">
          {source.doc_name}
          {source.used_in_context && <span className="badge-used">in prompt</span>}
        </div>
        <div className="source-preview">{source.text_preview}</div>
      </div>
      <div className="source-right">
        <div className="source-score">{(source.score * 100).toFixed(1)}%</div>
        <div className="source-expand">read ↗</div>
      </div>
    </div>
  )
}

function AnswerCard({ result }) {
  const [activeSource, setActiveSource] = useState(null)
  const usedCount = result.sources.filter(s => s.used_in_context).length

  return (
    <>
      <div className="answer-card">
        <div className="answer-header">
          <span>🤖 Answer</span>
        </div>
        <div className="answer-body">{result.answer}</div>
        <div className="metrics-row">
          <span className="metric">⏱ Retrieval: <strong>{result.retrieval_time_ms.toFixed(0)} ms</strong></span>
          <span className="metric">⚡ Generation: <strong>{result.generation_time_s.toFixed(1)} s</strong></span>
          <span className="metric">📄 Used in prompt: <strong>{usedCount}</strong></span>
          <span className="metric">🔍 Retrieved: <strong>{result.sources.length}</strong></span>
        </div>

        {result.sources.length > 0 && (
          <div className="sources-section">
            <div className="sources-title">
              📚 Sources Retrieved
              <span className="sources-hint">Click any to read full text</span>
            </div>
            <div className="source-cards">
              {result.sources.map((s, i) => (
                <SourceCard
                  key={i}
                  source={s}
                  rank={i + 1}
                  onClick={() => setActiveSource(s)}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {activeSource && (
        <ChunkModal source={activeSource} onClose={() => setActiveSource(null)} />
      )}
    </>
  )
}

export default function App() {
  const status = useStatus()
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [topK, setTopK] = useState(5)
  const [busy, setBusy] = useState(false)
  const bottomRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const submit = useCallback(async (question) => {
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
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Server error')
      }
      const result = await res.json()
      setMessages(prev => prev.map((m, i) => i === idx ? { ...m, result, loading: false } : m))
    } catch (e) {
      setMessages(prev => prev.map((m, i) => i === idx
        ? { ...m, result: { answer: `⚠️ Error: ${e.message}`, sources: [], retrieval_time_ms: 0, generation_time_s: 0, num_chunks: 0 }, loading: false }
        : m))
    } finally {
      setBusy(false)
    }
  }, [input, busy, status, topK, messages.length])

  const onKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="logo">🔬</div>
          <div>
            <h1>RAG QA System</h1>
            <p>ML Research Paper Q&amp;A · Variant 5 · AAI-2501</p>
          </div>
        </div>
        <StatusPill status={status} />
      </header>

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
              <div className="user-bubble">{m.q}</div>
              {m.loading ? <Thinking /> : <AnswerCard result={m.result} />}
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </main>

      <div className="input-bar">
        <div className="topk-row">
          <span>Sources to retrieve: <strong>{topK}</strong></span>
          <input type="range" min={1} max={10} value={topK} onChange={e => setTopK(+e.target.value)} />
          <span className="small">(all shown, top {topK > 2 ? 2 : topK} used in prompt)</span>
        </div>
        <div className="input-wrap">
          <textarea
            ref={textareaRef}
            rows={1}
            placeholder="Ask a question about ML papers… (Enter to send)"
            value={input}
            onChange={e => {
              setInput(e.target.value)
              e.target.style.height = 'auto'
              e.target.style.height = e.target.scrollHeight + 'px'
            }}
            onKeyDown={onKey}
            disabled={busy || status.status !== 'ready'}
          />
          <button
            className="send-btn"
            onClick={() => submit()}
            disabled={busy || !input.trim() || status.status !== 'ready'}
          >
            {busy ? '⏳' : '➤'}
          </button>
        </div>
        <div className="input-hint">Enter to send · Shift+Enter for new line</div>
      </div>
    </div>
  )
}
