import { useState } from 'react'
import { api } from './api'
import { useApi } from './ui'
import Overview from './pages/Overview'
import Segments from './pages/Segments'
import Leaderboard from './pages/Leaderboard'
import Customer from './pages/Customer'
import Batch from './pages/Batch'
import Insights from './pages/Insights'
import Assistant from './pages/Assistant'

const NAV = [
  ['overview', 'Overview'],
  ['segments', 'Segments'],
  ['leaderboard', 'Risk leaderboard'],
  ['customer', 'Customer explorer'],
  ['batch', 'Score new data'],
  ['insights', 'Model insights'],
  ['assistant', 'Ask the analyst'],
]

export default function App() {
  const [view, setView] = useState('overview')
  const [customerId, setCustomerId] = useState('')
  const health = useApi(() => api.health(), [])
  const openCustomer = (id) => { setCustomerId(String(id)); setView('customer') }

  return (
    <div className="flex h-full">
      <aside className="flex w-60 shrink-0 flex-col bg-ink-950 p-5 text-white">
        <div className="mb-8">
          <div className="font-display text-xl font-semibold leading-tight">Vantara</div>
          <div className="label mt-1 !text-ink-300">Customer behaviour</div>
        </div>
        <nav className="flex flex-1 flex-col gap-1" aria-label="Main">
          {NAV.map(([key, label]) => (
            <button key={key} onClick={() => setView(key)} aria-current={view === key ? 'page' : undefined}
              className={`rounded-lg px-3 py-2 text-left text-sm transition ${view === key ? 'bg-white/10 font-semibold text-white' : 'text-ink-300 hover:bg-white/5 hover:text-white'}`}>
              {label}
            </button>
          ))}
        </nav>
        <div className="mt-4 rounded-lg bg-white/5 p-3 text-xs text-ink-300">
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${health.data?.status === 'ok' ? 'bg-risk-low' : health.loading ? 'bg-risk-med' : 'bg-risk-high'}`} />
            {health.loading ? 'Checking API…' : health.data?.status === 'ok' ? 'API healthy' : 'API unavailable'}
          </div>
          {health.data && <div className="mt-1 num">{health.data.scored_customers.toLocaleString()} customers scored</div>}
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto p-8">
        {health.error || health.data?.status === 'degraded' ? (
          <div role="alert" className="card mb-6 border-risk-high/40 p-5 text-sm">
            <b className="text-risk-high">The backend isn’t ready.</b> Run <code className="font-mono">python -m src.pipeline</code> once to train the models and populate the database, then start the API with <code className="font-mono">uvicorn api.main:app</code>.
          </div>
        ) : null}
        {view === 'overview' && <Overview openCustomer={openCustomer} go={setView} />}
        {view === 'segments' && <Segments />}
        {view === 'leaderboard' && <Leaderboard openCustomer={openCustomer} />}
        {view === 'customer' && <Customer initialId={customerId} />}
        {view === 'batch' && <Batch />}
        {view === 'insights' && <Insights />}
        {view === 'assistant' && <Assistant />}
      </main>
    </div>
  )
}
