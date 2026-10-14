// Thin fetch wrapper: JSON in/out, readable errors (the API returns {detail, errors}).
async function request(path, options = {}) {
  let res
  try {
    res = await fetch(path, options)
  } catch {
    throw new Error('Cannot reach the API. Is the backend running on port 8000?')
  }
  if (!res.ok) {
    let msg = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (body.errors) msg = body.errors.map((e) => `${e.field}: ${e.problem}`).join('; ')
      else if (body.detail) msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* non-JSON error body */ }
    throw new Error(msg)
  }
  return res.json()
}

const qs = (params) => {
  const p = new URLSearchParams()
  Object.entries(params || {}).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== '') p.set(k, v) })
  const s = p.toString()
  return s ? `?${s}` : ''
}

export const api = {
  health: () => request('/api/health'),
  overview: () => request('/api/overview'),
  filters: () => request('/api/filters'),
  revenue: () => request('/api/revenue'),
  segments: (f) => request(`/api/segments${qs(f)}`),
  segmentPoints: (f) => request(`/api/segments/points${qs(f)}`),
  leaderboard: (f) => request(`/api/customers/leaderboard${qs(f)}`),
  customer: (id) => request(`/api/customers/${id}`),
  explain: (id, f) => request(`/api/customers/${id}/explain${qs(f)}`),
  predict: (body, narrative) => request(`/api/predict${qs({ narrative })}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  batch: (file) => { const fd = new FormData(); fd.append('file', file); return request('/api/predict/batch', { method: 'POST', body: fd }) },
  metadata: () => request('/api/model/metadata'),
  globalExplain: () => request('/api/explain/global'),
  ollamaStatus: () => request('/api/ollama/status'),
  ask: (question) => request('/api/assistant/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }) }),
}
