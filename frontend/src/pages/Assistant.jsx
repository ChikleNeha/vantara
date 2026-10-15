import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, ArrowUp, Bot, Lightbulb, PhoneCall, RotateCcw, Sparkles, TrendingDown, User, Users } from 'lucide-react'
import { api } from '../api'
import { useApi } from '../ui'

const IDEAS = [
  ['Which segment has the most spend at risk?', Users, 'tone-blue'],
  ['What are the main reasons customers leave?', Lightbulb, 'tone-sky'],
  ['How many customers are high risk?', TrendingDown, 'tone-peri'],
  ['Which customers should we call first?', PhoneCall, 'tone-mist'],
]
const stamp = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

function Avatar({ user, error }) {
  const Icon = user ? User : error ? AlertTriangle : Bot
  const tone = user ? 'bg-ink-950 text-white' : error ? 'bg-risk-high/10 text-risk-high' : 'bg-pastel-blue text-ink-950'
  return <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full ${tone}`}><Icon className="h-[18px] w-[18px]" /></div>
}

export default function Assistant() {
  const status = useApi(() => api.ollamaStatus(), [])
  const [msgs, setMsgs] = useState([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const scroller = useRef(null)
  const box = useRef(null)

  useEffect(() => { scroller.current?.scrollTo?.({ top: scroller.current.scrollHeight, behavior: 'smooth' }) }, [msgs, busy])
  useEffect(() => {
    const el = box.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`
  }, [input])

  const send = async (text) => {
    const q = (text ?? input).trim()
    if (q.length < 3 || busy) return
    const id = () => `${Date.now()}-${Math.random()}`
    setInput('')
    setMsgs((m) => [...m, { id: id(), role: 'user', text: q, time: stamp() }])
    setBusy(true)
    try {
      const r = await api.ask(q)
      setMsgs((m) => [...m, { id: id(), role: 'bot', text: r.answer, time: stamp() }])
    } catch (e) {
      setMsgs((m) => [...m, { id: id(), role: 'error', text: e.message, time: stamp() }])
    } finally { setBusy(false) }
  }
  const onKey = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }

  const s = status.data
  const online = s?.available && s?.model_pulled
  const statusText = status.loading ? 'Checking…' : online ? `Online · ${s.model}` : s?.available ? `Model “${s.model}” not pulled` : 'Offline · start Ollama to get answers'

  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col">
      <div className="card flex min-h-0 flex-1 flex-col overflow-hidden">
        <header className="flex items-center gap-3 border-b border-paper-line px-6 py-4">
          <Avatar />
          <div className="min-w-0 flex-1">
            <div className="font-display text-base font-bold leading-tight">Vantara Analyst</div>
            <div className="flex items-center gap-1.5 text-xs text-ink-500">
              <span className={`h-2 w-2 rounded-full ${online ? 'bg-risk-low' : status.loading ? 'bg-risk-med' : 'bg-risk-high'}`} />
              {statusText}
            </div>
          </div>
          <button className="btn-ghost" onClick={() => setMsgs([])} disabled={!msgs.length || busy}><RotateCcw className="h-4 w-4" />New chat</button>
        </header>

        <div ref={scroller} className="flex-1 space-y-5 overflow-y-auto px-6 py-6" aria-live="polite">
          {msgs.length === 0 && (
            <div className="mx-auto flex max-w-2xl flex-col items-center pt-6 text-center">
              <div className="float mb-5 flex h-16 w-16 items-center justify-center rounded-3xl bg-pastel-blue text-ink-950"><Sparkles className="h-8 w-8" /></div>
              <h2 className="text-3xl font-bold text-ink-950">What would you like to know?</h2>
              <p className="mt-2 max-w-md text-sm text-ink-500">Ask about segments, churn risk or revenue. Answers come from a local model that only sees a summary of your scored customers.</p>
              <div className="stagger mt-8 grid w-full gap-3 sm:grid-cols-2">
                {IDEAS.map(([text, Icon, tone]) => (
                  <button key={text} onClick={() => send(text)} className={`lift flex items-start gap-3 rounded-3xl p-5 text-left ${tone}`}>
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-white text-ink-950"><Icon className="h-[18px] w-[18px]" /></span>
                    <span className="text-sm font-semibold leading-snug text-ink-950">{text}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {msgs.map((m) => m.role === 'user' ? (
            <div key={m.id} className="msg-in flex items-end justify-end gap-3">
              <div className="max-w-[75%]">
                <div className="whitespace-pre-wrap rounded-3xl rounded-br-md bg-ink-950 px-5 py-3 text-[15px] leading-relaxed text-white">{m.text}</div>
                <div className="mt-1 pr-2 text-right text-[11px] text-ink-500">{m.time}</div>
              </div>
              <Avatar user />
            </div>
          ) : (
            <div key={m.id} className="msg-in flex items-end gap-3">
              <Avatar error={m.role === 'error'} />
              <div className="max-w-[75%]">
                <div className={`whitespace-pre-wrap rounded-3xl rounded-bl-md px-5 py-3 text-[15px] leading-relaxed ${m.role === 'error' ? 'bg-risk-high/10 text-risk-high' : 'bg-pastel-sky text-ink-950'}`}>{m.text}</div>
                <div className="mt-1 pl-2 text-[11px] text-ink-500">{m.time}</div>
              </div>
            </div>
          ))}

          {busy && (
            <div className="msg-in flex items-end gap-3" role="status" aria-label="Analyst is typing">
              <Avatar />
              <div className="flex items-center gap-1.5 rounded-3xl rounded-bl-md bg-pastel-sky px-5 py-4"><span className="dot" /><span className="dot" /><span className="dot" /></div>
            </div>
          )}
        </div>

        <footer className="border-t border-paper-line bg-white p-4">
          <div className="flex items-end gap-2 rounded-3xl bg-paper px-4 py-2 transition focus-within:ring-4 focus-within:ring-accent-soft">
            <textarea ref={box} rows={1} value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={onKey} maxLength={500} aria-label="Question"
              placeholder="Message the analyst…" className="max-h-36 flex-1 resize-none bg-transparent py-2 text-[15px] outline-none placeholder:text-ink-500" />
            <button onClick={() => send()} disabled={busy || input.trim().length < 3} aria-label="Send"
              className="mb-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-ink-950 text-white transition duration-200 hover:-translate-y-0.5 hover:bg-accent-deep active:scale-95 disabled:bg-ink-300 disabled:hover:translate-y-0">
              <ArrowUp className="h-5 w-5" />
            </button>
          </div>
          <p className="mt-2 text-center text-[11px] text-ink-500">Enter to send · Shift+Enter for a new line · the model explains numbers, it never predicts</p>
        </footer>
      </div>
    </div>
  )
}
