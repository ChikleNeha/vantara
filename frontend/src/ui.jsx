import { useEffect, useState, useCallback } from 'react'

export const money = (n) => (n == null || Number.isNaN(n) ? '—' : '£' + new Intl.NumberFormat('en-GB', { maximumFractionDigits: 0 }).format(n))
export const pct = (n, d = 0) => (n == null ? '—' : `${(n * 100).toFixed(d)}%`)
export const int = (n) => (n == null ? '—' : new Intl.NumberFormat('en-GB').format(Math.round(n)))
export const RISK_COLOR = { high: '#c2432b', medium: '#d39a1c', low: '#2e7d6a' }
export const SEG_COLORS = ['#2f5d8a', '#c2432b', '#2e7d6a', '#d39a1c', '#7a5aa6', '#3a8fa3', '#8a6d3b', '#b04a7a']

// Data hook: runs `fn` whenever deps change; exposes loading and error.
export function useApi(fn, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true })
  const run = useCallback(() => {
    let live = true
    setState((s) => ({ ...s, loading: true, error: null }))
    fn().then((data) => live && setState({ data, error: null, loading: false }))
        .catch((e) => live && setState({ data: null, error: e.message, loading: false }))
    return () => { live = false }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  useEffect(() => run(), [run])
  return state
}

export function PageHeader({ title, subtitle, children }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-3xl font-semibold text-ink-950">{title}</h1>
        {subtitle && <p className="mt-1 max-w-2xl text-sm text-ink-500">{subtitle}</p>}
      </div>
      <div className="no-print flex items-center gap-2">{children}</div>
    </div>
  )
}

export function Stat({ label, value, hint, tone }) {
  const color = tone === 'high' ? 'text-risk-high' : tone === 'low' ? 'text-risk-low' : 'text-ink-950'
  return (
    <div className="card p-4">
      <div className="label">{label}</div>
      <div className={`num mt-2 font-display text-3xl font-semibold ${color}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-ink-500">{hint}</div>}
    </div>
  )
}

export function Badge({ band }) {
  const map = { high: 'bg-risk-high/10 text-risk-high', medium: 'bg-risk-med/15 text-[#8a6410]', low: 'bg-risk-low/10 text-risk-low' }
  return <span className={`inline-flex rounded-full px-2 py-0.5 text-xs font-semibold capitalize ${map[band] || 'bg-paper text-ink-500'}`}>{band}</span>
}

export function Spinner({ label = 'Loading…' }) {
  return <div role="status" className="flex items-center gap-3 p-6 text-sm text-ink-500"><span className="h-4 w-4 animate-spin rounded-full border-2 border-ink-300 border-t-ink-900" />{label}</div>
}

export function ErrorBox({ message }) {
  return <div role="alert" className="rounded-lg border border-risk-high/30 bg-risk-high/5 p-4 text-sm text-risk-high">{message}</div>
}

export function Async({ state, children, label }) {
  if (state.loading) return <Spinner label={label} />
  if (state.error) return <ErrorBox message={state.error} />
  return children(state.data)
}

export function Select({ label, value, onChange, options, all = 'All' }) {
  return (
    <label className="block min-w-[150px]">
      <span className="label">{label}</span>
      <select className="field mt-1" value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">{all}</option>
        {options.map((o) => <option key={o} value={o}>{o}</option>)}
      </select>
    </label>
  )
}

export function downloadCsv(rows, filename) {
  if (!rows.length) return
  const cols = Object.keys(rows[0])
  const esc = (v) => { const s = v == null ? '' : String(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s }
  const csv = [cols.join(','), ...rows.map((r) => cols.map((c) => esc(r[c])).join(','))].join('\n')
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}
