import { useRef, useState } from 'react'
import { api } from '../api'
import { PageHeader, Spinner } from '../ui'

const IDEAS = ['Which segment has the most spend at risk?', 'What are the main reasons customers leave?', 'How many customers are high risk?', 'Which customers should we call first?']

export default function Assistant() {
  const [q, setQ] = useState('')
  const [log, setLog] = useState([])
  const [busy, setBusy] = useState(false)
  const end = useRef(null)

  const ask = async (text) => {
    const question = (text ?? q).trim()
    if (question.length < 3) return
    setQ(''); setBusy(true)
    setLog((l) => [...l, { role: 'user', text: question }])
    try {
      const r = await api.ask(question)
      setLog((l) => [...l, { role: 'bot', text: r.answer }])
    } catch (e) { setLog((l) => [...l, { role: 'error', text: e.message }]) } finally { setBusy(false); setTimeout(() => end.current?.scrollIntoView({ behavior: 'smooth' }), 50) }
  }

  return (
    <>
      <PageHeader title="Ask the analyst" subtitle="Questions about the current portfolio, answered by a local language model that only sees a summary of the scored data." />
      <div className="card flex h-[60vh] max-w-3xl flex-col">
        <div className="flex-1 space-y-3 overflow-y-auto p-5" aria-live="polite">
          {log.length === 0 && (
            <div><p className="mb-3 text-sm text-ink-500">Try one of these:</p>
              <div className="flex flex-wrap gap-2">{IDEAS.map((i) => <button key={i} className="btn-ghost" onClick={() => ask(i)}>{i}</button>)}</div></div>)}
          {log.map((m, i) => (
            <div key={i} className={`max-w-[85%] rounded-xl px-4 py-2.5 text-sm leading-relaxed ${m.role === 'user' ? 'ml-auto bg-ink-900 text-white' : m.role === 'error' ? 'border border-risk-high/30 bg-risk-high/5 text-risk-high' : 'border border-paper-line bg-white'}`}>{m.text}</div>
          ))}
          {busy && <Spinner label="Thinking…" />}
          <div ref={end} />
        </div>
        <form onSubmit={(e) => { e.preventDefault(); ask() }} className="flex gap-2 border-t border-paper-line p-3">
          <input className="field" placeholder="Ask about segments, risk, drivers…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Question" maxLength={500} />
          <button className="btn" disabled={busy}>Ask</button>
        </form>
      </div>
      <p className="mt-3 max-w-3xl text-xs text-ink-500">Requires Ollama running locally. The model never computes predictions; it only explains numbers produced by the trained models.</p>
    </>
  )
}
