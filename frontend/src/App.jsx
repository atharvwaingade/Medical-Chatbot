import { useMemo, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const severityColor = {
  low: '#16a34a',
  medium: '#d97706',
  high: '#dc2626',
}

const emptyResponse = {
  possible_conditions: [],
  explanation: '',
  severity: 'low',
  recommended_action: '',
  when_to_see_doctor: '',
  confidence: 'low',
  disclaimer: 'This is not medical advice',
}

function App() {
  const [query, setQuery] = useState('')
  const [symptoms, setSymptoms] = useState('')
  const [response, setResponse] = useState(emptyResponse)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const severityStyle = useMemo(
    () => ({ color: severityColor[response.severity] || severityColor.low }),
    [response.severity],
  )

  const saveHistory = (label, data) => {
    setHistory((prev) => [{ label, data }, ...prev].slice(0, 8))
  }

  const sendAsk = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      if (!res.ok) throw new Error('Unable to process request')
      const body = await res.json()
      setResponse(body)
      saveHistory(query, body)
      setQuery('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const sendSymptomCheck = async () => {
    const list = symptoms
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
    if (!list.length) return

    setLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/symptom-check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symptoms: list }),
      })
      if (!res.ok) throw new Error('Unable to process symptom check')
      const body = await res.json()
      setResponse(body)
      saveHistory(`Symptoms: ${list.join(', ')}`, body)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app">
      <header>
        <h1>Personal Medical Assistant</h1>
        <p className="disclaimer-banner">
          ⚠️ This assistant is for informational use only and does not provide diagnosis.
        </p>
      </header>

      <section className="panel">
        <h2>Chat</h2>
        <div className="row">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a health question..."
          />
          <button onClick={sendAsk} disabled={loading}>
            {loading ? 'Checking...' : 'Ask'}
          </button>
        </div>
      </section>

      <section className="panel">
        <h2>Symptom Checker</h2>
        <div className="row">
          <input
            value={symptoms}
            onChange={(e) => setSymptoms(e.target.value)}
            placeholder="Enter symptoms separated by commas"
          />
          <button onClick={sendSymptomCheck} disabled={loading}>
            Check
          </button>
        </div>
      </section>

      {error ? <p className="error">{error}</p> : null}

      <section className="panel response">
        <h2>Result</h2>
        <p>
          <strong>Possible conditions:</strong> {response.possible_conditions.join(', ') || 'N/A'}
        </p>
        <p>
          <strong>Explanation:</strong> {response.explanation || 'N/A'}
        </p>
        <p>
          <strong>Severity:</strong>{' '}
          <span style={severityStyle} className="severity">
            {response.severity}
          </span>
        </p>
        <p>
          <strong>Recommended action:</strong> {response.recommended_action || 'N/A'}
        </p>
        <p>
          <strong>When to see doctor:</strong> {response.when_to_see_doctor || 'N/A'}
        </p>
        <p>
          <strong>Confidence:</strong> {response.confidence}
        </p>
        <p className="disclaimer">{response.disclaimer}</p>
      </section>

      <section className="panel history">
        <h2>History</h2>
        {history.length === 0 ? (
          <p>No history yet.</p>
        ) : (
          <ul>
            {history.map((item, idx) => (
              <li key={`${item.label}-${idx}`}>
                <strong>{item.label}</strong>
                <div>{item.data.severity} severity • {item.data.confidence} confidence</div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </main>
  )
}

export default App
