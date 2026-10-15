import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'
import fx from './__fixtures__/api.json'
import App from './App.jsx'

// Serve real API responses (captured from the running backend) for every endpoint.
function mockFetch() {
  globalThis.fetch = vi.fn(async (url, opts = {}) => {
    const u = String(url)
    const json = (body, status = 200) => ({ ok: status < 400, status, json: async () => body })
    if (u.startsWith('/api/health')) return json(fx.health)
    if (u.startsWith('/api/overview')) return json(fx.overview)
    if (u.startsWith('/api/filters')) return json(fx.filters)
    if (u.startsWith('/api/revenue')) return json(fx.revenue)
    if (u.startsWith('/api/segments/points')) return json(fx.segment_points)
    if (u.startsWith('/api/segments')) return json(fx.segments)
    if (u.startsWith('/api/customers/leaderboard')) return json(fx.leaderboard)
    if (u.startsWith('/api/explain/global')) return json(fx.global)
    if (u.includes('/explain')) return json(u.includes('lime') ? fx.lime : fx.explain)
    if (/\/api\/customers\/\d+/.test(u)) return json(fx.customer)
    if (u.startsWith('/api/predict/batch')) return json({ summary: { mode: 'transactions', customers_scored: 1, high_risk: 1, flagged_churn: 1, avg_churn_prob: 0.8, spend_at_risk_90d: 100 }, cleaning_report: {}, rows: fx.leaderboard.rows.slice(0, 1).map((r) => ({ ...r, segment_label: r.segment_label })) })
    if (u.startsWith('/api/predict')) return json(fx.predict)
    if (u.startsWith('/api/model/metadata')) return json(fx.metadata)
    if (u.startsWith('/api/ollama/status')) return json(fx.ollama)
    if (u.startsWith('/api/assistant/ask')) return json({ detail: 'The local language model is not reachable' }, 503)
    return json({ detail: 'not found' }, 404)
  })
}

const nav = (name) => fireEvent.click(screen.getByRole('button', { name }))

describe('dashboard against real API payloads', () => {
  beforeEach(() => { cleanup(); mockFetch() })

  it('overview shows KPIs, forecast chart and leaderboard preview', async () => {
    render(<App />)
    expect(await screen.findByText('Customers scored')).toBeTruthy()
    expect(await screen.findByText(/Highest expected spend at risk/)).toBeTruthy()
    expect(screen.getByText(/synthetic demo data/)).toBeTruthy()
  })

  it('segments page renders every segment card', async () => {
    render(<App />); nav('Segments')
    for (const p of fx.segments.profiles) expect(await screen.findByText(p.label)).toBeTruthy()
  })

  it('leaderboard lists customers and opens the explorer on click', async () => {
    render(<App />); nav('Risk leaderboard')
    const id = String(fx.leaderboard.rows[0].customer_id)
    const cell = await screen.findByText(id)
    fireEvent.click(cell)
    expect(await screen.findByText(/In plain language/)).toBeTruthy()
    expect(screen.getByText(/What drives this prediction/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Run LIME' }))
    await waitFor(() => expect(screen.getByText(fx.lime.lime[0].rule)).toBeTruthy())
  })

  it('what-if scorer returns a prediction', async () => {
    render(<App />); nav('Score new data')
    fireEvent.click(await screen.findByRole('button', { name: 'Score customer' }))
    expect(await screen.findByText(/predicted next 90d spend/)).toBeTruthy()
  })

  it('model insights shows the comparison table and global importance', async () => {
    render(<App />); nav('Model insights')
    expect(await screen.findByText(/Churn models/)).toBeTruthy()
    expect(await screen.findByText(/What matters most overall/)).toBeTruthy()
  })

  it('assistant surfaces a helpful message when Ollama is down', async () => {
    render(<App />); nav('Ask the analyst')
    fireEvent.click(screen.getByRole('button', { name: /How many customers are high risk/ }))
    expect(await screen.findByText(/not reachable/)).toBeTruthy()
  })

  it('chat sends on Enter, shows a typing state, and New chat clears it', async () => {
    render(<App />); nav('Ask the analyst')
    const box = screen.getByLabelText('Question')
    fireEvent.change(box, { target: { value: 'Who is at risk?' } })
    fireEvent.keyDown(box, { key: 'Enter', shiftKey: false })
    expect(await screen.findByText('Who is at risk?')).toBeTruthy()
    expect(await screen.findByText(/not reachable/)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /New chat/ }))
    await waitFor(() => expect(screen.queryByText('Who is at risk?')).toBeNull())
    expect(screen.getByText('What would you like to know?')).toBeTruthy()
  })
})
