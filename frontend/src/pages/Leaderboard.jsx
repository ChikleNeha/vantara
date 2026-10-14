import { useState } from 'react'
import { api } from '../api'
import { Async, Badge, PageHeader, Select, downloadCsv, int, money, pct, useApi } from '../ui'

const PAGE = 25

export default function Leaderboard({ openCustomer }) {
  const [f, setF] = useState({ segment: '', country: '', value_tier: '', risk_band: '' })
  const [page, setPage] = useState(0)
  const set = (k) => (v) => { setF((s) => ({ ...s, [k]: v })); setPage(0) }
  const filters = useApi(() => api.filters(), [])
  const lb = useApi(() => api.leaderboard({ ...f, limit: PAGE, offset: page * PAGE }), [f, page])

  const exportAll = async () => { const all = await api.leaderboard({ ...f, limit: 200 }); downloadCsv(all.rows, 'risk_leaderboard.csv') }

  return (
    <>
      <PageHeader title="Churn risk leaderboard" subtitle="Ranked by expected spend at risk, so valuable customers who are likely to leave appear first.">
        <button className="btn-ghost" onClick={exportAll}>Export top 200 (CSV)</button>
      </PageHeader>
      <div className="card no-print mb-5 flex flex-wrap gap-4 p-4">
        <Select label="Segment" value={f.segment} onChange={set('segment')} options={filters.data?.segments || []} />
        <Select label="Country" value={f.country} onChange={set('country')} options={filters.data?.countries || []} />
        <Select label="Value tier" value={f.value_tier} onChange={set('value_tier')} options={filters.data?.value_tiers || []} />
        <Select label="Risk" value={f.risk_band} onChange={set('risk_band')} options={filters.data?.risk_bands || []} />
      </div>
      <div className="card overflow-hidden">
        <Async state={lb}>
          {(d) => (
            <>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead><tr>{['#', 'Customer', 'Segment', 'Country', 'Value', 'Risk', 'Churn prob.', 'Usual 90d spend', 'Spend at risk', 'Likely next category'].map((h) => <th key={h} className="th">{h}</th>)}</tr></thead>
                  <tbody>
                    {d.rows.map((r, i) => (
                      <tr key={r.customer_id} onClick={() => openCustomer(r.customer_id)} className="cursor-pointer hover:bg-paper">
                        <td className="td num text-ink-500">{page * PAGE + i + 1}</td>
                        <td className="td font-mono">{r.customer_id}{r.is_anomaly ? <span title="Unusual behaviour" className="ml-1 text-risk-med">●</span> : null}</td>
                        <td className="td">{r.segment_label}</td><td className="td">{r.country}</td>
                        <td className="td capitalize">{r.value_tier}</td><td className="td"><Badge band={r.risk_band} /></td>
                        <td className="td num">{pct(r.churn_prob)}</td><td className="td num">{money(r.avg_spend_90d)}</td>
                        <td className="td num font-semibold">{money(r.priority_score)}</td><td className="td capitalize">{(r.next_category || '—').replace('_', ' ')}</td>
                      </tr>
                    ))}
                    {d.rows.length === 0 && <tr><td className="td text-ink-500" colSpan={10}>No customers match these filters.</td></tr>}
                  </tbody>
                </table>
              </div>
              <div className="no-print flex items-center justify-between border-t border-paper-line p-3 text-sm">
                <span className="text-ink-500 num">{int(d.total)} customers · page {page + 1} of {Math.max(1, Math.ceil(d.total / PAGE))}</span>
                <div className="flex gap-2">
                  <button className="btn-ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
                  <button className="btn-ghost" disabled={(page + 1) * PAGE >= d.total} onClick={() => setPage(page + 1)}>Next</button>
                </div>
              </div>
            </>
          )}
        </Async>
      </div>
      <p className="mt-3 text-xs text-ink-500">● marks accounts the autoencoder considers unusual. Click a row for the full explanation.</p>
    </>
  )
}
