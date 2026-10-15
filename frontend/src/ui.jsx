import { useEffect, useState, useCallback } from 'react'

export const money = (n) => (n == null || Number.isNaN(n) ? '—' : '£' + new Intl.NumberFormat('en-GB', { maximumFractionDigits: 0 }).format(n))
export const pct = (n, d = 0) => (n == null ? '—' : `${(n * 100).toFixed(d)}%`)
export const int = (n) => (n == null ? '—' : new Intl.NumberFormat('en-GB').format(Math.round(n)))
export const RISK_COLOR = { high: '#c2432b', medium: '#d39a1c', low: '#2e7d6a' }
export const SEG_COLORS = ['#4f8cff', '#ff7a59', '#2fb38a', '#f5b83d', '#8b6cf0', '#35a7c9', '#c0728a', '#6f7bb8']

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
        <h1 className="text-4xl font-bold text-ink-950">{title}</h1>
        {subtitle && <p className="mt-1 max-w-2xl text-sm text-ink-500">{subtitle}</p>}
      </div>
      <div className="no-print flex items-center gap-2">{children}</div>
    </div>
  )
}

const BG = { white: 'bg-white', blue: 'tone-blue', sky: 'tone-sky', peri: 'tone-peri', mist: 'tone-mist' }

// Counts a number up from 0 once it arrives; skipped for reduced-motion users and non-browser test envs.
export function useCountUp(target, ms = 900) {
  const [v, setV] = useState(0)
  useEffect(() => {
    if (target == null) return undefined
    const still = typeof requestAnimationFrame === 'undefined' || window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches
    if (still) { setV(target); return undefined }
    let raf, start
    const tick = (t) => {
      start ??= t
      const p = Math.min(1, (t - start) / ms)
      setV(target * (1 - Math.pow(1 - p, 3)))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, ms])
  return v
}

export function Stat({ label, value, num, fmt = int, hint, tone, bg = 'white' }) {
  const n = useCountUp(num)
  const color = tone === 'high' ? 'text-risk-high' : tone === 'low' ? 'text-risk-low' : 'text-ink-950'
  return (
    <div className={`card lift p-6 ${BG[bg]}`}>
      <span className="label">{label}</span>
      <div className={`num mt-6 font-display text-4xl font-bold ${color}`}>{num != null ? fmt(n) : value}</div>
      {hint && <div className="mt-2 text-sm text-ink-500">{hint}</div>}
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
      <span className="text-xs font-semibold text-ink-500">{label}</span>
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
