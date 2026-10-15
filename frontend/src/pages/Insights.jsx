import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from 'recharts'
import { api } from '../api'
import { Async, PageHeader, pct, useApi } from '../ui'

const f3 = (v) => (v == null ? '—' : Number(v).toFixed(3))

export default function Insights() {
  const meta = useApi(() => api.metadata(), [])
  const glob = useApi(() => api.globalExplain(), [])
  const oll = useApi(() => api.ollamaStatus(), [])
  return (
    <>
      <PageHeader title="Model insights" subtitle="How the models compare, what they rely on, and which one is serving predictions." />
      <Async state={meta}>
        {(m) => (
          <>
            <div className="card mb-6 overflow-x-auto p-7">
              <div className="label mb-1">Churn models (test set)</div>
              <p className="mb-3 mt-2 text-xs text-ink-500">Ranked by ROC-AUC, then recall. In production: <b className="text-ink-900">{m.production_model}</b> (threshold {m.threshold.toFixed(3)}), chosen on validation data. Trained {m.trained_at}.</p>
              <table className="w-full"><thead><tr>{['Model', 'CV AUC', 'Test AUC', 'Recall', 'Precision', 'F1', 'Accuracy'].map((h) => <th key={h} className="th">{h}</th>)}</tr></thead>
                <tbody>{m.comparison_table.map((r) => (
                  <tr key={r.model} className={r.model === m.production_model ? 'bg-accent-soft/50' : ''}>
                    <td className="td font-medium">{r.model}{r.model === m.production_model ? ' ★' : ''}</td><td className="td num">{f3(r.cv_auc)}</td><td className="td num">{f3(r.roc_auc)}</td>
                    <td className="td num">{f3(r.recall)}</td><td className="td num">{f3(r.precision)}</td><td className="td num">{f3(r.f1)}</td><td className="td num">{f3(r.accuracy)}</td></tr>))}</tbody></table>
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="card p-7">
                <div className="label mb-3">Lifetime value (next 90 days)</div>
                <table className="w-full"><thead><tr>{['Model', 'Val R²', 'Test R²', 'MAE'].map((h) => <th key={h} className="th">{h}</th>)}</tr></thead>
                  <tbody>{Object.entries(m.clv).filter(([k]) => k !== 'best').map(([k, v]) => (
                    <tr key={k} className={k === m.clv.best ? 'bg-accent-soft/50' : ''}><td className="td">{k}{k === m.clv.best ? ' ★' : ''}</td><td className="td num">{f3(v.val_r2)}</td><td className="td num">{f3(v.raw.r2)}</td><td className="td num">{v.raw.mae.toFixed(0)}</td></tr>))}</tbody></table>
              </div>
              <div className="card p-7">
                <div className="label mb-3">Platform status</div>
                <dl className="grid grid-cols-2 gap-y-2 text-sm">
                  <dt className="text-ink-500">Data source</dt><dd className="text-right capitalize">{m.data_source}</dd>
                  <dt className="text-ink-500">Latency (model + SHAP)</dt><dd className="num text-right">p50 {m.latency_ms?.p50} ms · p95 {m.latency_ms?.p95} ms</dd>
                  <dt className="text-ink-500">Segment quality</dt><dd className="num text-right">silhouette {m.segmentation_quality?.kmeans?.silhouette}</dd>
                  <dt className="text-ink-500">Neural net (test AUC)</dt><dd className="num text-right">{f3(m.deep?.ann?.test?.roc_auc)}</dd>
                  <dt className="text-ink-500">LSTM (test AUC)</dt><dd className="num text-right">{f3(m.deep?.lstm?.test?.roc_auc)}</dd>
                  <dt className="text-ink-500">Ollama</dt>
                  <dd className="text-right">{oll.data ? (oll.data.available ? (oll.data.model_pulled ? `ready (${oll.data.model})` : `running, model “${oll.data.model}” not pulled`) : 'not running (templates used)') : '…'}</dd>
                </dl>
              </div>
            </div>
          </>
        )}
      </Async>
      <Async state={glob}>
        {(g) => (
          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <div className="card p-7">
              <div className="label mb-1">What matters most overall (mean |SHAP|)</div>
              {g.shap_lime_top5_agreement != null && <p className="mb-3 text-xs text-ink-500">SHAP and LIME agree on {pct(g.shap_lime_top5_agreement)} of top-5 factors per customer.</p>}
              {g.importance.slice(0, 10).map((d) => {
                const max = g.importance[0].importance
                return (
                  <div key={d.feature} className="mb-2"><div className="flex justify-between text-sm"><span className="capitalize">{d.label}</span><span className="num text-ink-500">{d.importance.toFixed(3)}</span></div>
                    <div className="h-2 rounded-full bg-paper"><div className="h-full rounded-full bg-accent bar-grow" style={{ width: `${(d.importance / max) * 100}%` }} /></div></div>)
              })}
            </div>
            <div className="card p-7">
              <div className="label mb-3">How churn risk changes with the top factors</div>
              {Object.entries(g.pdp).map(([feat, c]) => {
                const label = g.importance.find((i) => i.feature === feat)?.label || feat
                const data = c.x.map((x, i) => ({ x, y: c.y[i] }))
                return (
                  <div key={feat} className="mb-3"><div className="text-xs capitalize text-ink-500">{label}</div>
                    <ResponsiveContainer width="100%" height={110}><LineChart data={data} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
                      <CartesianGrid stroke="#e4ded2" vertical={false} /><XAxis dataKey="x" tick={{ fontSize: 10 }} /><YAxis domain={[0, 1]} tick={{ fontSize: 10 }} width={30} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
                      <Tooltip formatter={(v) => pct(v, 1)} /><Line dataKey="y" stroke="#c2432b" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></div>)
              })}
            </div>
          </div>
        )}
      </Async>
    </>
  )
}
