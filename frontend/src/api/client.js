const BASE = '/api'

async function fetchJSON(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options)
  const data = await res.json()
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`)
  return data
}

export const api = {
  getTeams: () => fetchJSON('/teams'),

  predict: (homeTeam, awayTeam) =>
    fetchJSON('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ home_team: homeTeam, away_team: awayTeam }),
    }),

  getTeamStats: (teamName) =>
    fetchJSON(`/team/${encodeURIComponent(teamName)}/stats`),

  getHealth: () => fetchJSON('/health'),
}
