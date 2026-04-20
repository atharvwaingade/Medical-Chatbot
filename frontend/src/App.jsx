import { useEffect, useRef, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const SEVERITY_COLOR = { low: '#16a34a', medium: '#d97706', high: '#dc2626' }
const CONFIDENCE_COLOR = { low: '#dc2626', medium: '#d97706', high: '#16a34a' }
const CONFIDENCE_WIDTH = { low: '30%', medium: '65%', high: '100%' }

const EMPTY_RESPONSE = {
  possible_conditions: [],
  explanation: '',
  severity: 'low',
  recommended_action: '',
  when_to_see_doctor: '',
  confidence: 'low',
  disclaimer: '',
  sources: [],
}

const HISTORY_KEY = 'medical_assistant_history'

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]')
  } catch {
    return []
  }
}

function saveHistory(history) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history))
  } catch {
    /* storage full — silently ignore */
  }
}

function SeverityBadge({ value }) {
  return (
    <span className="badge" style={{ background: SEVERITY_COLOR[value] || SEVERITY_COLOR.low }}>
      {value}
    </span>
  )
}

function ConfidenceBar({ value }) {
  return (
    <div className="confidence-track" title={`Confidence: ${value}`}>
      <div
        className="confidence-fill"
        style={{
          width: CONFIDENCE_WIDTH[value] || CONFIDENCE_WIDTH.low,
          background: CONFIDENCE_COLOR[value] || CONFIDENCE_COLOR.low,
        }}
      />
    </div>
  )
}

function App() {
  const [query, setQuery] = useState('')
  const [symptoms, setSymptoms] = useState('')
  const [response, setResponse] = useState(EMPTY_RESPONSE)
  const [history, setHistory] = useState(loadHistory)
  const [loading, setLoading] = useState(false)
  const [activeOp, setActiveOp] = useState(null) // 'ask' | 'check'
  const [error, setError] = useState('')
  const resultRef = useRef(null)

  // Persist history to localStorage whenever it changes
  useEffect(() => {
    saveHistory(history)
  }, [history])

  const addToHistory = (label, data) => {
    setHistory((prev) => [{ label, data, ts: Date.now() }, ...prev].slice(0, 10))
  }

  const scrollToResult = () => {
    setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
  }

  const sendAsk = async () => {
    if (!query.trim() || loading) return
    setLoading(true)
    setActiveOp('ask')
    setError('')
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim() }),
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const body = await res.json()
      setResponse(body)
      addToHistory(query.trim(), body)
      setQuery('')
      scrollToResult()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setActiveOp(null)
    }
  }

  const sendSymptomCheck = async () => {
    const list = symptoms
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
    if (!list.length || loading) return
    setLoading(true)
    setActiveOp('check')
    setError('')
    try {
      const res = await fetch(`${API_BASE}/symptom-check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: list }),
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const body = await res.json()
      setResponse(body)
      addToHistory(`Symptoms: ${list.join(', ')}`, body)
      scrollToResult()
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setActiveOp(null)
    }
  }

  const clearHistory = () => {
    setHistory([])
    localStorage.removeItem(HISTORY_KEY)
  }

  const hasResult = response.possible_conditions.length > 0

  return (
    <main className="app">
      <header>
        <h1>🩺 Personal Medical Assistant</h1>
        <p className="disclaimer-banner">
          ⚠️ For informational use only. Not a substitute for professional medical advice,
          diagnosis, or treatment.
        </p>
      </header>

      {/* ── Ask panel ──────────────────────────────────────────────── */}
      <section className="panel">
        <h2>Ask a health question</h2>
        <div className="row">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendAsk()}
            placeholder="e.g. What could cause a persistent cough and fever?"
            disabled={loading}
            aria-label="Health question"
          />
          <button onClick={sendAsk} disabled={loading || !query.trim()} className="btn-primary">
            {activeOp === 'ask' ? <span className="spinner" /> : 'Ask'}
          </button>
        </div>
        <p className="hint">Press Enter or click Ask</p>
      </section>

      {/* ── Symptom checker panel ───────────────────────────────────── */}
      <section className="panel">
        <h2>Symptom checker</h2>
        <div className="row">
          <input
            value={symptoms}
            onChange={(e) => setSymptoms(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendSymptomCheck()}
            placeholder="fever, headache, fatigue  (comma-separated)"
            disabled={loading}
            aria-label="Symptom list"
          />
          <button
            onClick={sendSymptomCheck}
            disabled={loading || !symptoms.trim()}
            className="btn-primary"
          >
            {activeOp === 'check' ? <span className="spinner" /> : 'Check'}
          </button>
        </div>
        <p className="hint">Press Enter or click Check</p>
      </section>

      {error && <p className="error" role="alert">{error}</p>}

      {/* ── Result panel ────────────────────────────────────────────── */}
      {hasResult && (
        <section className="panel response" ref={resultRef}>
          <h2>Result</h2>

          <div className="result-row">
            <strong>Possible conditions:</strong>
            <span className="conditions">
              {response.possible_conditions.join(', ')}
            </span>
          </div>

          <div className="result-row">
            <strong>Explanation:</strong>
            <span>{response.explanation}</span>
          </div>

          <div className="result-row">
            <strong>Severity:</strong>
            <SeverityBadge value={response.severity} />
          </div>

          <div className="result-row">
            <strong>Confidence:</strong>
            <div className="confidence-wrapper">
              <span className="confidence-label">{response.confidence}</span>
              <ConfidenceBar value={response.confidence} />
            </div>
          </div>

          <div className="result-row">
            <strong>Recommended action:</strong>
            <span>{response.recommended_action}</span>
          </div>

          <div className="result-row">
            <strong>When to see a doctor:</strong>
            <span>{response.when_to_see_doctor}</span>
          </div>

          {response.sources && response.sources.length > 0 && (
            <div className="result-row sources">
              <strong>Sources:</strong>
              <ul className="source-list">
                {response.sources.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </div>
          )}

          <p className="disclaimer">{response.disclaimer}</p>
        </section>
      )}

      {/* ── History panel ───────────────────────────────────────────── */}
      <section className="panel history">
        <div className="history-header">
          <h2>History</h2>
          {history.length > 0 && (
            <button onClick={clearHistory} className="btn-ghost btn-small">
              Clear
            </button>
          )}
        </div>

        {history.length === 0 ? (
          <p className="muted">No history yet. Ask a question or check symptoms above.</p>
        ) : (
          <ul>
            {history.map((item, idx) => (
              <li
                key={`${item.ts}-${idx}`}
                className="history-item"
                onClick={() => setResponse(item.data)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === 'Enter' && setResponse(item.data)}
                title="Click to restore this result"
              >
                <div className="history-label">{item.label}</div>
                <div className="history-meta">
                  <SeverityBadge value={item.data.severity} />
                  <span className="muted">confidence: {item.data.confidence}</span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  )
}

export default App
