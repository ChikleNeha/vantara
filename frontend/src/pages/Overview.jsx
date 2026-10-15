import { ComposedChart, Line, Area, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from 'recharts'
import { api } from '../api'
import { Async, Badge, PageHeader, int, money, pct, useApi, useCountUp } from '../ui'

// Overview keeps the classic card look: white, thin border, small radius, compact type.
const CARD = 'rounded-xl border border-paper-line bg-white shadow-card'
const TAG = 'text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-500'

// Compact currency (e.g. £1.2M) so large totals never overflow a KPI card; the exact value is in the tooltip.
const moneyCompact = (n) => '£' + new Intl.NumberFormat('en-GB', { notation: 'compact', maximumFractionDigits: 1 }).format(n)

function Kpi({ label, num, fmt = int, exact, hint, tone }) {
  const n = useCountUp(num)
  const color = tone === 'high' ? 'text-risk-high' : 'text-ink-950'
  return (
    <div className={`${CARD} min-w-0 p-4`}>
      <div className={`${TAG} truncate`} title={label}>{label}</div>
      <div className={`num mt-2 truncate font-display text-2xl font-semibold ${color}`} title={exact ?? fmt(num)}>{fmt(n)}</div>
      {hint && <div className="mt-1 truncate text-xs text-ink-500" title={hint}>{hint}</div>}
    </div>
  )
}

function RevenueChart({ rows }) {
  const lastActual = [...rows].reverse().find((r) => !r.is_forecast && !r.is_partial)?.month
  const data = rows.map((r) => ({
    month: r.month,
    actual: !r.is_forecast && !r.is_partial ? r.revenue : null,
    partial: r.is_partial ? r.revenue : null,
    forecast: r.is_forecast || r.month === lastActual ? r.revenue : null,
  }))
  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={data} margin={{ left: 8, right: 8, top: 8 }}>
        <CartesianGrid stroke="#e6e9f4" vertical={false} />
        <XAxis dataKey="month" tick={{ fontSize: 11 }} interval={2} />
        <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => `£${Math.round(v / 1000)}k`} width={52} />
        <Tooltip formatter={(v, n) => [money(v), n === 'partial' ? 'Partial month' : n[0].toUpperCase() + n.slice(1)]} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Area type="monotone" dataKey="actual" name="Actual" stroke="#4f8cff" fill="#4f8cff" fillOpacity={0.12} strokeWidth={2} connectNulls />
        <Line type="monotone" dataKey="forecast" name="Forecast" stroke="#c2432b" strokeDasharray="6 4" strokeWidth={2} dot={false} connectNulls />
        <Line type="monotone" dataKey="partial" name="Partial month" stroke="#d39a1c" strokeWidth={0} dot={{ r: 5, fill: '#d39a1c' }} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

function RiskBars({ bands, total }) {
  const items = [['high', 'bg-risk-high'], ['medium', 'bg-risk-med'], ['low', 'bg-risk-low']]
  return (
    <div className="space-y-4">
      {items.map(([k, cls]) => (
        <div key={k}>
          <div className="mb-1 flex justify-between text-sm"><span className="capitalize">{k} risk</span><span className="num text-ink-500">{int(bands[k] || 0)} · {pct((bands[k] || 0) / total)}</span></div>
          <div className="h-2.5 overflow-hidden rounded-full bg-paper"><div className={`h-full bar-grow ${cls}`} style={{ width: `${((bands[k] || 0) / total) * 100}%` }} /></div>
        </div>
      ))}
    </div>
  )
}

export default function Overview({ openCustomer, go }) {
  const ov = useApi(() => api.overview(), [])
  const rev = useApi(() => api.revenue(), [])
  const top = useApi(() => api.leaderboard({ limit: 6 }), [])
  return (
    <>
      <PageHeader title="Where to focus this quarter" subtitle="Customers most likely to stop buying in the next 90 days, weighted by how much they normally spend." />
      <Async state={ov}>
        {(d) => (
          <>
            {d.data_source === 'synthetic' && (
              <div role="note" className="mb-5 rounded-lg border border-risk-med/40 bg-risk-med/10 p-3 text-sm text-[#7a5a10]">
                Showing <b>synthetic demo data</b>. Put <code className="font-mono">online_retail_II.xlsx</code> in <code className="font-mono">data/raw/</code> and re-run the pipeline to see real results.
              </div>
            )}
            <div className="stagger grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-5">
              <Kpi label="Customers scored" num={d.kpis.customers} />
              <Kpi label="High risk" num={d.kpis.high_risk} tone="high" hint={pct(d.kpis.high_risk / d.kpis.customers) + ' of base'} />
              <Kpi label="Spend at risk (90d)" num={d.kpis.spend_at_risk} fmt={moneyCompact} exact={money(d.kpis.spend_at_risk)} tone="high" hint="churn prob. × usual spend" />
              <Kpi label="Avg churn probability" num={d.kpis.avg_churn_prob} fmt={pct} />
              <Kpi label="Unusual accounts" num={d.kpis.anomalies} hint="flagged by autoencoder" />
            </div>
            <div className="mt-6 grid gap-4 lg:grid-cols-3">
              <div className={`${CARD} p-5 lg:col-span-2`}>
                <div className={`${TAG} mb-3`}>Monthly revenue and forecast</div>
                <Async state={rev}>{(r) => <><RevenueChart rows={r.rows} /><p className="mt-2 text-xs text-ink-500">{r.method}. Dashed line is the forecast.</p></>}</Async>
              </div>
              <div className={`${CARD} p-5`}>
                <div className={`${TAG} mb-4`}>Risk distribution</div>
                <RiskBars bands={d.risk_bands} total={d.kpis.customers} />
                <div className="mt-6 border-t border-paper-line pt-4 text-xs text-ink-500">
                  Model: <b className="text-ink-900">{d.production_model}</b> · test AUC <span className="num">{d.test_auc}</span> · recall <span className="num">{d.test_recall}</span>
                </div>
              </div>
            </div>
          </>
        )}
      </Async>
      <div className={`${CARD} mt-6 p-5`}>
        <div className="mb-3 flex items-center justify-between">
          <div className={TAG}>Highest expected spend at risk</div>
          <button className="btn-ghost no-print" onClick={() => go('leaderboard')}>See all</button>
        </div>
        <Async state={top}>
          {(t) => (
            <table className="w-full"><thead><tr>{['Customer', 'Segment', 'Risk', 'Churn prob.', 'Spend at risk'].map((h) => <th key={h} className="th">{h}</th>)}</tr></thead>
              <tbody>{t.rows.map((r) => (
                <tr key={r.customer_id} className="cursor-pointer hover:bg-paper" onClick={() => openCustomer(r.customer_id)}>
                  <td className="td font-mono">{r.customer_id}</td><td className="td">{r.segment_label}</td><td className="td"><Badge band={r.risk_band} /></td>
                  <td className="td num">{pct(r.churn_prob)}</td><td className="td num">{money(r.priority_score)}</td>
                </tr>))}</tbody></table>
          )}
        </Async>
      </div>
    </>
  )
}