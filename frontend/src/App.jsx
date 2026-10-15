import { useState } from 'react'
import { BarChart3, Layers, LayoutDashboard, MessageCircle, Search, Sparkles, TrendingDown, UploadCloud } from 'lucide-react'
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
  ['overview', 'Overview', LayoutDashboard],
  ['segments', 'Segments', Layers],
  ['leaderboard', 'Risk leaderboard', TrendingDown],
  ['customer', 'Customer explorer', Search],
  ['batch', 'Score new data', UploadCloud],
  ['insights', 'Model insights', BarChart3],
  ['assistant', 'Ask the analyst', MessageCircle],
]

export default function App() {
  const [view, setView] = useState('overview')
  const [customerId, setCustomerId] = useState('')
  const health = useApi(() => api.health(), [])
  const openCustomer = (id) => { setCustomerId(String(id)); setView('customer') }

  return (
    <div className="flex h-full">
      <aside className="flex w-64 shrink-0 flex-col bg-ink-950 p-5 text-white">
        <div className="mb-9 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-pastel-blue text-ink-950"><Sparkles className="h-5 w-5" /></div>
          <div>
            <div className="font-display text-lg font-bold leading-tight">Vantara</div>
            <div className="text-xs text-ink-300">Customer behaviour</div>
          </div>
        </div>
        <nav className="flex flex-1 flex-col gap-1.5" aria-label="Main">
          {NAV.map(([key, label, Icon]) => (
            <button key={key} onClick={() => setView(key)} aria-current={view === key ? 'page' : undefined}
              className={`group flex items-center gap-3 rounded-2xl px-3.5 py-2.5 text-left text-sm transition-all duration-200 ${view === key ? 'bg-pastel-blue font-semibold text-ink-950 shadow-lg shadow-pastel-blue/20' : 'text-ink-300 hover:bg-white/10 hover:text-white'}`}>
              <Icon className="h-[18px] w-[18px] shrink-0 transition-transform duration-200 group-hover:scale-110" />
              {label}
            </button>
          ))}
        </nav>
        <div className="mt-4 rounded-2xl bg-white/5 p-3.5 text-xs text-ink-300">
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${health.data?.status === 'ok' ? 'bg-risk-low' : health.loading ? 'bg-risk-med' : 'bg-risk-high'}`} />
            {health.loading ? 'Checking API…' : health.data?.status === 'ok' ? 'API healthy' : 'API unavailable'}
          </div>
          {health.data && <div className="num mt-1">{health.data.scored_customers.toLocaleString()} customers scored</div>}
        </div>
      </aside>
      <main className="flex-1 overflow-y-auto p-8">
        {health.error || health.data?.status === 'degraded' ? (
          <div role="alert" className="card mb-6 bg-risk-high/5 p-5 text-sm">
            <b className="text-risk-high">The backend isn’t ready.</b> Run <code className="font-mono">python -m src.pipeline</code> once to train the models and populate the database, then start the API with <code className="font-mono">uvicorn api.main:app</code>.
          </div>
        ) : null}
        <div key={view} className="page-enter">
          {view === 'overview' && <Overview openCustomer={openCustomer} go={setView} />}
          {view === 'segments' && <Segments />}
          {view === 'leaderboard' && <Leaderboard openCustomer={openCustomer} />}
          {view === 'customer' && <Customer initialId={customerId} />}
          {view === 'batch' && <Batch />}
          {view === 'insights' && <Insights />}
          {view === 'assistant' && <Assistant />}
        </div>
      </main>
    </div>
  )
}
