import { useRef, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const SEVERITY_COLOR = { low: '#16a34a', medium: '#d97706', high: '#dc2626' }
const CONFIDENCE_COLOR = { low: '#dc2626', medium: '#d97706', high: '#16a34a' }
const CONFIDENCE_WIDTH = { low: '30%', medium: '65%', high: '100%' }
const EVIDENCE_TIER_COLOR = { 1: '#1e40af', 2: '#2563eb', 3: '#0369a1', 4: '#6b7280', 5: '#9ca3af' }
const EVIDENCE_TIER_LABEL = {
  1: 'Meta-analysis',
  2: 'RCT',
  3: 'Guideline',
  4: 'Peer-reviewed',
  5: 'Consensus',
}
const PREVALENCE_COLOR = {
  'very common': '#16a34a',
  common: '#2563eb',
  uncommon: '#d97706',
  rare: '#dc2626',
}

const EMPTY_RESPONSE = {
  possible_conditions: [],
  explanation: '',
  severity: 'low',
  recommended_action: '',
  when_to_see_doctor: '',
  confidence: 'low',
  disclaimer: '',
  sources: [],
  session_id: null,
  retrieval_metadata: null,
}

// History is kept in React state only — not persisted to browser storage to
// avoid clear-text storage of potentially sensitive medical query data.
function loadHistory() { return [] }

// ---------------------------------------------------------------------------
// Reusable UI components
// ---------------------------------------------------------------------------

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

function EvidenceBadge({ tier }) {
  return (
    <span
      className="evidence-badge"
      style={{ background: EVIDENCE_TIER_COLOR[tier] || '#9ca3af' }}
      title={`Evidence Tier ${tier}`}
    >
      T{tier} {EVIDENCE_TIER_LABEL[tier] || 'Unknown'}
    </span>
  )
}

function PrevalenceBadge({ value }) {
  return (
    <span
      className="prevalence-badge"
      style={{ color: PREVALENCE_COLOR[value] || '#6b7280' }}
    >
      ◉ {value}
    </span>
  )
}

function MetadataPanel({ meta, className = '' }) {
  if (!meta) return null
  const entropy = typeof meta.retrieval_entropy === 'number'
    ? meta.retrieval_entropy.toFixed(3)
    : '—'
  const hasNegated = meta.negated_terms?.length > 0
  const hasExpanded = meta.expanded_terms?.length > 0
  return (
    <div className={`metadata-panel ${className}`}>
      <span className="metadata-title">🔬 Retrieval Metadata</span>
      <span className="metadata-chip chip-neutral">
        Entropy: {entropy}
      </span>
      <span className="metadata-chip chip-neutral">
        {meta.retriever_type || 'MedHybrid-BM25+SCS+PRF'}
      </span>
      {hasNegated && (
        <span className="metadata-chip chip-warning" title="NegEx-detected negations excluded from retrieval">
          ⊘ Negated: {meta.negated_terms.slice(0, 4).join(', ')}
        </span>
      )}
      {hasExpanded && (
        <span className="metadata-chip chip-info" title="RM3-PRF synonym expansion">
          ↗ Expanded: {meta.expanded_terms.slice(0, 4).join(', ')}
        </span>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Differential Diagnosis Table
// ---------------------------------------------------------------------------

function DifferentialTable({ differentials }) {
  if (!differentials?.length) return null
  return (
    <div className="diff-table-wrapper">
      <table className="diff-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Condition</th>
            <th>ICD-10</th>
            <th>Score</th>
            <th>Symptom Match</th>
            <th>Evidence</th>
            <th>Prevalence</th>
            <th>Severity</th>
            <th>Ruling In ✓</th>
            <th>Ruling Out ✗</th>
          </tr>
        </thead>
        <tbody>
          {differentials.map((d) => (
            <tr key={d.rank} className={d.rank === 1 ? 'diff-row-top' : ''}>
              <td className="diff-rank">{d.rank}</td>
              <td className="diff-condition">
                <strong>{d.condition}</strong>
                <br />
                <span className="muted" style={{ fontSize: '0.75rem' }}>{d.source}</span>
              </td>
              <td className="diff-icd10">{d.icd10 || '—'}</td>
              <td className="diff-score">{(d.hybrid_score * 100).toFixed(1)}%</td>
              <td>
                <div className="scs-bar-track">
                  <div
                    className="scs-bar-fill"
                    style={{ width: `${Math.round(d.symptom_match_ratio * 100)}%` }}
                  />
                </div>
                <span className="muted" style={{ fontSize: '0.72rem' }}>
                  {Math.round(d.symptom_match_ratio * 100)}%
                </span>
              </td>
              <td><EvidenceBadge tier={d.evidence_tier} /></td>
              <td><PrevalenceBadge value={d.prevalence} /></td>
              <td><SeverityBadge value={d.severity} /></td>
              <td className="diff-symptoms ruling-in">
                {d.ruling_in_symptoms?.length > 0
                  ? d.ruling_in_symptoms.map((s) => (
                    <span key={s} className="sym-chip chip-in">{s}</span>
                  ))
                  : <span className="muted">—</span>}
              </td>
              <td className="diff-symptoms ruling-out">
                {d.ruling_out_symptoms?.length > 0
                  ? d.ruling_out_symptoms.slice(0, 3).map((s) => (
                    <span key={s} className="sym-chip chip-out">{s}</span>
                  ))
                  : <span className="muted">—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

function App() {
  const [activeTab, setActiveTab] = useState('ask')   // 'ask' | 'differential'
  const [query, setQuery] = useState('')
  const [symptoms, setSymptoms] = useState('')
  const [response, setResponse] = useState(EMPTY_RESPONSE)
  const [differential, setDifferential] = useState(null)
  const [history, setHistory] = useState(loadHistory)
  const [loading, setLoading] = useState(false)
  const [activeOp, setActiveOp] = useState(null)
  const [error, setError] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const resultRef = useRef(null)
  const diffRef = useRef(null)

  const addToHistory = (label, data) => {
    setHistory((prev) => [{ label, data, ts: Date.now() }, ...prev].slice(0, 10))
  }

  const scrollTo = (ref) => {
    setTimeout(() => ref.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 50)
  }

  const sendAsk = async () => {
    if (!query.trim() || loading) return
    setLoading(true); setActiveOp('ask'); setError('')
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: query.trim(), session_id: sessionId }),
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const body = await res.json()
      setResponse(body)
      if (body.session_id) setSessionId(body.session_id)
      addToHistory(query.trim(), body)
      setQuery('')
      scrollTo(resultRef)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false); setActiveOp(null)
    }
  }

  const sendSymptomCheck = async () => {
    const list = symptoms.split(',').map((s) => s.trim()).filter(Boolean)
    if (!list.length || loading) return
    setLoading(true); setActiveOp('check'); setError('')
    try {
      const res = await fetch(`${API_BASE}/symptom-check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: list, session_id: sessionId }),
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const body = await res.json()
      setResponse(body)
      if (body.session_id) setSessionId(body.session_id)
      addToHistory(`Symptoms: ${list.join(', ')}`, body)
      scrollTo(resultRef)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false); setActiveOp(null)
    }
  }

  const sendDifferential = async () => {
    const list = symptoms.split(',').map((s) => s.trim()).filter(Boolean)
    if (!list.length || loading) return
    setLoading(true); setActiveOp('diff'); setError('')
    try {
      const res = await fetch(`${API_BASE}/differential`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: list, query: query.trim(), session_id: sessionId }),
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const body = await res.json()
      setDifferential(body)
      if (body.session_id) setSessionId(body.session_id)
      scrollTo(diffRef)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false); setActiveOp(null)
    }
  }

  const clearHistory = () => { setHistory([]) }

  const resetSession = () => {
    setSessionId(null)
    setResponse(EMPTY_RESPONSE)
    setDifferential(null)
  }

  const hasResult = response.possible_conditions.length > 0

  return (
    <main className="app">
      <header>
        <div className="header-top">
          <h1>🩺 Personal Medical Assistant</h1>
          <div className="header-badges">
            <span className="arch-badge">MedHybrid-BM25+SCS+PRF</span>
            {sessionId && (
              <button onClick={resetSession} className="btn-ghost btn-tiny" title="Clear conversation session">
                ⟳ New session
              </button>
            )}
          </div>
        </div>
        <p className="disclaimer-banner">
          ⚠️ For informational use only. Not a substitute for professional medical advice,
          diagnosis, or treatment.
        </p>
      </header>

      {/* ── Tab navigation ─────────────────────────────────────────── */}
      <div className="tab-nav">
        <button
          className={`tab-btn ${activeTab === 'ask' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('ask')}
        >
          Ask / Symptom Check
        </button>
        <button
          className={`tab-btn ${activeTab === 'differential' ? 'tab-active' : ''}`}
          onClick={() => setActiveTab('differential')}
        >
          Differential Diagnosis
        </button>
      </div>

      {/* ── Ask tab ────────────────────────────────────────────────── */}
      {activeTab === 'ask' && (
        <>
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
            <p className="hint">
              Press Enter or click Ask
              {sessionId && <span className="session-chip">🔗 Session active</span>}
            </p>
          </section>

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

          {hasResult && (
            <section className="panel response" ref={resultRef}>
              <h2>Result</h2>

              <div className="result-row">
                <strong>Possible conditions:</strong>
                <span className="conditions">{response.possible_conditions.join(', ')}</span>
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
              {response.sources?.length > 0 && (
                <div className="result-row sources">
                  <strong>Sources:</strong>
                  <ul className="source-list">
                    {response.sources.map((s) => <li key={s}>{s}</li>)}
                  </ul>
                </div>
              )}

              <MetadataPanel meta={response.retrieval_metadata} className="meta-inline" />

              <p className="disclaimer">{response.disclaimer}</p>
            </section>
          )}
        </>
      )}

      {/* ── Differential tab ───────────────────────────────────────── */}
      {activeTab === 'differential' && (
        <>
          <section className="panel">
            <h2>Structured Differential Diagnosis</h2>
            <p className="hint" style={{ marginBottom: '0.75rem' }}>
              Enter symptoms below to see a ranked differential with evidence tiers,
              ICD-10 codes, and ruling-in/out symptoms.
            </p>
            <div className="row">
              <input
                value={symptoms}
                onChange={(e) => setSymptoms(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && sendDifferential()}
                placeholder="fever, cough, fatigue  (comma-separated)"
                disabled={loading}
                aria-label="Symptoms for differential"
              />
              <button
                onClick={sendDifferential}
                disabled={loading || !symptoms.trim()}
                className="btn-primary"
              >
                {activeOp === 'diff' ? <span className="spinner" /> : 'Analyse'}
              </button>
            </div>
            <div className="row" style={{ marginTop: '0.5rem' }}>
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Optional: add free-text context (e.g. 'started 3 days ago')"
                disabled={loading}
                aria-label="Optional context"
              />
            </div>
          </section>

          {error && <p className="error" role="alert">{error}</p>}

          {differential && (
            <section className="panel" ref={diffRef}>
              <h2>Differential Analysis</h2>

              {differential.query_metadata && (
                <MetadataPanel meta={differential.query_metadata} />
              )}

              <DifferentialTable differentials={differential.differentials} />

              <p className="disclaimer" style={{ marginTop: '1rem' }}>
                {differential.disclaimer}
              </p>
            </section>
          )}
        </>
      )}

      {/* ── History panel ──────────────────────────────────────────── */}
      <section className="panel history">
        <div className="history-header">
          <h2>History</h2>
          {history.length > 0 && (
            <button onClick={clearHistory} className="btn-ghost btn-small">Clear</button>
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
                onClick={() => { setResponse(item.data); setActiveTab('ask') }}
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
