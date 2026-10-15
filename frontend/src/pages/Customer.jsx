import { useEffect, useState } from 'react'
import { api } from '../api'
import { Badge, ErrorBox, PageHeader, Spinner, int, money, pct } from '../ui'

export function DriverBars({ drivers }) {
  const max = Math.max(...drivers.map((d) => Math.abs(d.shap)), 0.001)
  return (
    <div className="space-y-3" aria-label="Factors behind the prediction">
      {drivers.map((d) => {
        const w = (Math.abs(d.shap) / max) * 50
        const up = d.shap > 0
        return (
          <div key={d.feature}>
            <div className="mb-1 flex justify-between text-sm"><span className="capitalize">{d.label}</span><span className="num text-ink-500">{d.value_text}</span></div>
            <div className="relative h-3 rounded-full bg-paper">
              <div className="absolute left-1/2 top-0 h-full w-px bg-ink-300" />
              <div className={`bar-grow absolute top-0 h-full rounded-full ${up ? 'bg-risk-high' : 'bg-risk-low'}`} style={up ? { left: '50%', width: `${w}%`, transformOrigin: 'left' } : { right: '50%', width: `${w}%`, transformOrigin: 'right' }} />
            </div>
            <div className="mt-0.5 text-[11px] text-ink-500">{up ? 'pushes churn risk up' : 'pulls churn risk down'}</div>
          </div>
        )
      })}
    </div>
  )
}

export default function Customer({ initialId = '' }) {
  const [input, setInput] = useState(initialId)
  const [id, setId] = useState(initialId)
  const [detail, setDetail] = useState(null)
  const [exp, setExp] = useState(null)
  const [lime, setLime] = useState(null)
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)
  const [llmBusy, setLlmBusy] = useState(false)

  const load = async (cid) => {
    if (!cid) return
    setBusy(true); setErr(null); setLime(null)
    try {
      const [d, e] = await Promise.all([api.customer(cid), api.explain(cid)])
      setDetail(d); setExp(e)
    } catch (e) { setErr(e.message); setDetail(null); setExp(null) } finally { setBusy(false) }
  }
  useEffect(() => { if (initialId) { setInput(initialId); setId(initialId); load(initialId) } }, [initialId])

  const submit = (e) => { e.preventDefault(); const v = input.trim(); if (v) { setId(v); load(v) } }
  const askLlm = async () => {
    setLlmBusy(true)
    try { setExp(await api.explain(id, { narrative: 'llm' })) } catch (e) { setErr(e.message) } finally { setLlmBusy(false) }
  }
  const runLime = async () => {
    setLlmBusy(true)
    try { setLime((await api.explain(id, { method: 'lime' })).lime) } catch (e) { setErr(e.message) } finally { setLlmBusy(false) }
  }
  const p = detail?.profile

  return (
    <>
      <PageHeader title="Customer explorer" subtitle="Look up any customer and see why the model rates them the way it does." />
      <form onSubmit={submit} className="no-print mb-5 flex max-w-md gap-2">
        <input className="field font-mono" placeholder="Customer ID, e.g. 12345" value={input} onChange={(e) => setInput(e.target.value)} aria-label="Customer ID" inputMode="numeric" />
        <button className="btn" disabled={busy}>Search</button>
      </form>
      {busy && <Spinner />}
      {err && <ErrorBox message={err} />}
      {p && exp && !busy && (
        <div className="grid gap-4 lg:grid-cols-3">
          <div className="card p-7 lg:col-span-1">
            <div className="label">Customer {p.customer_id}</div>
            <div className="mt-2 flex items-center gap-3"><span className="num font-display text-5xl font-semibold">{pct(p.churn_prob)}</span><Badge band={p.risk_band} /></div>
            <p className="mt-1 text-xs text-ink-500">chance of no purchase in the next 90 days (flag threshold {pct(detail.threshold)})</p>
            <dl className="num mt-5 grid grid-cols-2 gap-y-2 text-sm">
              <dt className="text-ink-500">Segment</dt><dd className="text-right">{p.segment_label}</dd>
              <dt className="text-ink-500">Country</dt><dd className="text-right">{p.country}</dd>
              <dt className="text-ink-500">Orders</dt><dd className="text-right">{int(p.frequency)}</dd>
              <dt className="text-ink-500">Total spend</dt><dd className="text-right">{money(p.monetary_total)}</dd>
              <dt className="text-ink-500">Usual 90d spend</dt><dd className="text-right">{money(p.avg_spend_90d)}</dd>
              <dt className="text-ink-500">Spend at risk</dt><dd className="text-right font-semibold">{money(p.priority_score)}</dd>
              <dt className="text-ink-500">Predicted next 90d</dt><dd className="text-right">{money(p.clv_pred_90d)}</dd>
              <dt className="text-ink-500">Likely next category</dt><dd className="text-right capitalize">{(p.next_category || '—').replace('_', ' ')}</dd>
              <dt className="text-ink-500">Engagement</dt><dd className="text-right">{p.engagement_score?.toFixed(0)}/100</dd>
              <dt className="text-ink-500">Neural-net check</dt><dd className="text-right">{pct(p.ann_prob)}</dd>
              <dt className="text-ink-500">Unusual account</dt><dd className="text-right">{p.is_anomaly ? 'Yes' : 'No'}</dd>
            </dl>
          </div>
          <div className="card p-7 lg:col-span-2">
            <div className="mb-2 flex items-center justify-between">
              <div className="label">In plain language</div>
              <button className="btn-ghost no-print" onClick={askLlm} disabled={llmBusy}>{llmBusy ? 'Thinking…' : 'Rewrite with Ollama'}</button>
            </div>
            <p className="text-[15px] leading-relaxed">{exp.narrative}</p>
            <p className="mt-2 text-[11px] text-ink-500">Source: {exp.narrative_source === 'ollama' ? 'local language model, grounded in the factors below' : 'built-in template (Ollama not used)'}</p>
            <div className="label mb-3 mt-6">What drives this prediction (SHAP)</div>
            <DriverBars drivers={exp.drivers} />
          </div>
          <div className="card p-7">
            <div className="label mb-3">Where they spend</div>
            {Object.keys(detail.category_affinity).length === 0 && <p className="text-sm text-ink-500">No category data.</p>}
            {Object.entries(detail.category_affinity).slice(0, 6).map(([k, v]) => (
              <div key={k} className="mb-2"><div className="flex justify-between text-sm"><span className="capitalize">{k.replace('_', ' ')}</span><span className="num text-ink-500">{pct(v)}</span></div>
                <div className="h-2 rounded-full bg-paper"><div className="h-full rounded-full bg-accent bar-grow" style={{ width: `${v * 100}%` }} /></div></div>
            ))}
          </div>
          <div className="card p-7 lg:col-span-2">
            <div className="mb-2 flex items-center justify-between">
              <div className="label">Cross-check with LIME</div>
              <button className="btn-ghost no-print" onClick={runLime} disabled={llmBusy}>Run LIME</button>
            </div>
            {!lime && <p className="text-sm text-ink-500">LIME fits a small local model around this customer. If it tells the same story as SHAP, the explanation is more trustworthy.</p>}
            {lime && <ul className="space-y-1 text-sm">{lime.map((r) => <li key={r.rule} className="flex justify-between border-b border-paper-line/70 py-1"><span className="font-mono text-xs">{r.rule}</span><span className={`num ${r.weight > 0 ? 'text-risk-high' : 'text-risk-low'}`}>{r.weight > 0 ? '+' : ''}{r.weight.toFixed(3)}</span></li>)}</ul>}
          </div>
          <div className="card overflow-x-auto p-7 lg:col-span-3">
            <div className="label mb-3">Recent order lines</div>
            <table className="w-full"><thead><tr>{['Date', 'Invoice', 'Item', 'Category', 'Qty', 'Price', 'Total'].map((h) => <th key={h} className="th">{h}</th>)}</tr></thead>
              <tbody>{detail.recent_lines.map((l, i) => (
                <tr key={i}><td className="td num">{l.invoice_date?.slice(0, 10)}</td><td className="td font-mono text-xs">{l.invoice}</td><td className="td">{l.description}</td>
                  <td className="td capitalize">{l.category?.replace('_', ' ')}</td><td className="td num">{l.quantity}</td><td className="td num">{l.price?.toFixed(2)}</td>
                  <td className={`td num ${l.is_return ? 'text-risk-high' : ''}`}>{l.line_total?.toFixed(2)}{l.is_return ? ' (return)' : ''}</td></tr>))}</tbody></table>
          </div>
        </div>
      )}
    </>
  )
}
