import { useState } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend } from 'recharts'
import { api } from '../api'
import { Async, PageHeader, Select, SEG_COLORS, int, money, pct, useApi } from '../ui'

export default function Segments() {
  const [country, setCountry] = useState('')
  const [tier, setTier] = useState('')
  const filters = useApi(() => api.filters(), [])
  const seg = useApi(() => api.segments({ country, value_tier: tier }), [country, tier])
  const pts = useApi(() => api.segmentPoints({ country, value_tier: tier }), [country, tier])
  const labels = seg.data ? seg.data.profiles.map((p) => p.label) : []
  const colorOf = (l) => SEG_COLORS[Math.max(0, labels.indexOf(l)) % SEG_COLORS.length]

  return (
    <>
      <PageHeader title="Customer segments" subtitle="Groups of customers who behave alike, with what each group needs from us." />
      <div className="card no-print mb-5 flex flex-wrap gap-4 p-4">
        <Select label="Country" value={country} onChange={setCountry} options={filters.data?.countries || []} />
        <Select label="Value tier" value={tier} onChange={setTier} options={filters.data?.value_tiers || []} />
      </div>
      <Async state={seg}>
        {(d) => (
          <>
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {d.profiles.map((p) => {
                const live = d.live.find((l) => l.label === p.label)
                return (
                  <div key={p.segment_id} className="card p-4" style={{ borderTop: `3px solid ${colorOf(p.label)}` }}>
                    <h3 className="text-lg font-semibold">{p.label}</h3>
                    <p className="mt-1 min-h-[3.5rem] text-xs text-ink-500">{p.description}</p>
                    <dl className="num mt-3 grid grid-cols-2 gap-y-1 text-sm">
                      <dt className="text-ink-500">Customers</dt><dd className="text-right">{live ? int(live.customers) : 0}</dd>
                      <dt className="text-ink-500">Avg churn prob.</dt><dd className="text-right">{live ? pct(live.avg_churn_prob) : '—'}</dd>
                      <dt className="text-ink-500">Spend at risk</dt><dd className="text-right">{live ? money(live.spend_at_risk) : '—'}</dd>
                      <dt className="text-ink-500">Last bought</dt><dd className="text-right">{int(p.recency_days)} d ago</dd>
                      <dt className="text-ink-500">Orders</dt><dd className="text-right">{p.frequency.toFixed(1)}</dd>
                      <dt className="text-ink-500">Return rate</dt><dd className="text-right">{pct(p.return_rate)}</dd>
                    </dl>
                  </div>
                )
              })}
            </div>
            <div className="card mt-6 p-5">
              <div className="label mb-1">Segment map</div>
              <p className="mb-3 text-xs text-ink-500">Each dot is a customer, projected to two dimensions (PCA). Closer dots behave more alike.</p>
              <Async state={pts}>
                {(p) => (
                  <ResponsiveContainer width="100%" height={380}>
                    <ScatterChart margin={{ left: 0, right: 12, top: 8, bottom: 8 }}>
                      <CartesianGrid stroke="#e4ded2" />
                      <XAxis type="number" dataKey="x" tick={{ fontSize: 11 }} name="Component 1" />
                      <YAxis type="number" dataKey="y" tick={{ fontSize: 11 }} name="Component 2" width={36} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ payload }) => payload?.[0] ? (
                        <div className="card px-3 py-2 text-xs"><b>{payload[0].payload.segment}</b><br />Customer {payload[0].payload.customer_id}</div>) : null} />
                      <Legend wrapperStyle={{ fontSize: 12 }} />
                      {labels.map((l) => <Scatter key={l} name={l} data={p.points.filter((x) => x.segment === l)} fill={colorOf(l)} fillOpacity={0.55} />)}
                    </ScatterChart>
                  </ResponsiveContainer>
                )}
              </Async>
            </div>
            <p className="mt-3 text-xs text-ink-500">Cluster quality — K-Means silhouette {d.quality?.kmeans?.silhouette}, Davies-Bouldin {d.quality?.kmeans?.davies_bouldin}; Gaussian mixture silhouette {d.quality?.gmm?.silhouette}.</p>
          </>
        )}
      </Async>
    </>
  )
}
