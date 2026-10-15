import { useRef, useState } from 'react'
import { UploadCloud } from 'lucide-react'
import { api } from '../api'
import { Badge, ErrorBox, PageHeader, Spinner, Stat, downloadCsv, int, money, pct } from '../ui'
import { DriverBars } from './Customer'

const FIELDS = [
  ['recency_days', 'Days since last purchase', 120], ['frequency', 'Number of orders', 4], ['monetary_total', 'Total spend (£)', 600],
  ['tenure_days', 'Days since first purchase', 400], ['return_rate', 'Return rate (0–1, optional)', ''], ['purchase_trend', 'Order trend (optional)', ''],
]

function WhatIf() {
  const [form, setForm] = useState(Object.fromEntries(FIELDS.map(([k, , d]) => [k, String(d)])))
  const [res, setRes] = useState(null)
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)
  const submit = async (e) => {
    e.preventDefault(); setBusy(true); setErr(null)
    const body = {}
    Object.entries(form).forEach(([k, v]) => { if (v !== '') body[k] = Number(v) })
    try { setRes(await api.predict(body)) } catch (x) { setErr(x.message); setRes(null) } finally { setBusy(false) }
  }
  return (
    <div className="card p-7">
      <div className="label mb-3">What-if: score a single customer</div>
      <form onSubmit={submit} className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {FIELDS.map(([k, label]) => (
          <label key={k} className="block"><span className="text-xs text-ink-500">{label}</span>
            <input className="field num mt-1" type="number" step="any" value={form[k]} onChange={(e) => setForm({ ...form, [k]: e.target.value })} /></label>
        ))}
        <div className="flex items-end"><button className="btn w-full" disabled={busy}>{busy ? 'Scoring…' : 'Score customer'}</button></div>
      </form>
      {err && <div className="mt-3"><ErrorBox message={err} /></div>}
      {res && (
        <div className="mt-5 grid gap-5 border-t border-paper-line pt-5 lg:grid-cols-2">
          <div>
            <div className="flex items-center gap-3"><span className="num font-display text-4xl font-semibold">{pct(res.churn_probability)}</span><Badge band={res.risk_band} /></div>
            <p className="mt-1 text-sm text-ink-500">Segment: {res.segment} · predicted next 90d spend {money(res.clv_pred_90d)}</p>
            <p className="mt-3 text-sm leading-relaxed">{res.narrative}</p>
          </div>
          <DriverBars drivers={res.drivers} />
        </div>
      )}
    </div>
  )
}

export default function Batch() {
  const [res, setRes] = useState(null)
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(false)
  const [name, setName] = useState('')
  const input = useRef(null)

  const upload = async (file) => {
    if (!file) return
    setBusy(true); setErr(null); setRes(null); setName(file.name)
    try { setRes(await api.batch(file)) } catch (e) { setErr(e.message) } finally { setBusy(false) }
  }
  const onDrop = (e) => { e.preventDefault(); upload(e.dataTransfer.files?.[0]) }

  return (
    <>
      <PageHeader title="Score new data" subtitle="Upload raw transactions (Online Retail II columns) or a table of customer summaries. Results can be downloaded as CSV or saved as PDF.">
        {res && <><button className="btn-ghost" onClick={() => downloadCsv(res.rows, 'scored_customers.csv')}>Download CSV</button><button className="btn-ghost" onClick={() => window.print()}>Save as PDF</button></>}
      </PageHeader>
      <div onDragOver={(e) => e.preventDefault()} onDrop={onDrop} className="no-print card tone-sky lift mb-6 flex flex-col items-center justify-center border-2 border-dashed border-pastel-peri p-10 text-center">
        <UploadCloud className="float mb-3 h-10 w-10 text-accent-deep" />
        <p className="text-sm text-ink-700">Drop a CSV here, or</p>
        <button className="btn mt-3" onClick={() => input.current?.click()} disabled={busy}>Choose file</button>
        <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(e) => upload(e.target.files?.[0])} aria-label="CSV file" />
        <p className="mt-3 max-w-xl text-xs text-ink-500">Transactions need: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country. Customer summaries need at least recency_days, frequency, monetary_total and tenure_days.</p>
      </div>
      {busy && <Spinner label={`Scoring ${name}…`} />}
      {err && <ErrorBox message={err} />}
      {res && (
        <>
          <div className="mb-5 grid grid-cols-2 gap-4 lg:grid-cols-4">
            <Stat label="Customers scored" value={int(res.summary.customers_scored)} hint={res.summary.mode.replace('_', ' ')} />
            <Stat label="High risk" value={int(res.summary.high_risk)} tone="high" />
            <Stat label="Flagged as likely to churn" value={int(res.summary.flagged_churn)} />
            <Stat label="Spend at risk (90d)" value={money(res.summary.spend_at_risk_90d)} tone="high" />
          </div>
          <div className="card overflow-x-auto">
            <table className="w-full"><thead><tr>{['Customer', 'Risk', 'Churn prob.', 'Segment', 'Value', 'Usual 90d spend', 'Spend at risk', 'Next category'].map((h) => <th key={h} className="th">{h}</th>)}</tr></thead>
              <tbody>{res.rows.slice(0, 200).map((r) => (
                <tr key={r.customer_id}><td className="td font-mono">{r.customer_id}</td><td className="td"><Badge band={r.risk_band} /></td><td className="td num">{pct(r.churn_prob)}</td>
                  <td className="td">{r.segment_label}</td><td className="td capitalize">{r.value_tier}</td><td className="td num">{money(r.avg_spend_90d)}</td>
                  <td className="td num font-semibold">{money(r.priority_score)}</td><td className="td capitalize">{(r.next_category || '—').replace('_', ' ')}</td></tr>))}</tbody></table>
            {res.rows.length > 200 && <p className="p-3 text-xs text-ink-500">Showing the top 200 of {int(res.rows.length)}; the CSV contains all rows.</p>}
          </div>
        </>
      )}
      <div className="no-print mt-6"><WhatIf /></div>
    </>
  )
}
