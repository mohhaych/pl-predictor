import { useState, useEffect } from 'react'
import { api } from '../api/client'

export default function FixtureSelector({ onPredict }) {
  const [teams, setTeams] = useState([])
  const [homeTeam, setHomeTeam] = useState('')
  const [awayTeam, setAwayTeam] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    api.getTeams()
      .then(setTeams)
      .catch(() => setError('Could not load teams. Is the backend running?'))
  }, [])

  const canPredict = homeTeam && awayTeam && homeTeam !== awayTeam && !loading

  async function handleSubmit(e) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const result = await api.predict(homeTeam, awayTeam)
      onPredict(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card selector-card">
      <h1 className="app-title">Premier League Match Predictor</h1>
      <p className="subtitle">Select two teams to get an ML-powered outcome prediction.</p>

      <form onSubmit={handleSubmit} className="selector-form">
        <div className="selector-row">
          <div className="field">
            <label htmlFor="home-team">Home Team</label>
            <select
              id="home-team"
              value={homeTeam}
              onChange={(e) => setHomeTeam(e.target.value)}
            >
              <option value="">Select team…</option>
              {teams.map((t) => (
                <option key={t.name} value={t.name}>{t.name}</option>
              ))}
            </select>
          </div>

          <div className="vs-badge">VS</div>

          <div className="field">
            <label htmlFor="away-team">Away Team</label>
            <select
              id="away-team"
              value={awayTeam}
              onChange={(e) => setAwayTeam(e.target.value)}
            >
              <option value="">Select team…</option>
              {teams
                .filter((t) => t.name !== homeTeam)
                .map((t) => (
                  <option key={t.name} value={t.name}>{t.name}</option>
                ))}
            </select>
          </div>
        </div>

        {error && <p className="error-msg">{error}</p>}

        <button
          type="submit"
          className="predict-btn"
          disabled={!canPredict}
          aria-busy={loading}
        >
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>
    </div>
  )
}
