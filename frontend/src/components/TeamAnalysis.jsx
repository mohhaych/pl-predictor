import { useState, useEffect } from 'react'
import { api } from '../api/client'

function FormBadge({ result }) {
  const cls = result === 'W' ? 'badge-win' : result === 'D' ? 'badge-draw' : 'badge-loss'
  return <span className={`form-badge ${cls}`}>{result}</span>
}

export default function TeamAnalysis({ teamName, onClose }) {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    setStats(null)
    setError('')
    api.getTeamStats(teamName)
      .then(setStats)
      .catch((err) => setError(err.message))
  }, [teamName])

  return (
    <div className="card analysis-card">
      <div className="analysis-header">
        <h2>{teamName}: Team Analysis</h2>
        <button className="back-btn" onClick={onClose} aria-label="Close team analysis">
          ← Back to Prediction
        </button>
      </div>

      {error && <p className="error-msg">{error}</p>}

      {!stats && !error && <p className="loading-msg">Loading…</p>}

      {stats && (
        <>
          <section aria-label="Recent form">
            <h3>Recent Form</h3>
            <div className="form-badges large">
              {stats.recent_form.map((m, i) => (
                <FormBadge key={i} result={m.result} />
              ))}
              {stats.recent_form.length === 0 && <p>No recent matches found.</p>}
            </div>
            <table className="form-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Opponent</th>
                  <th>H/A</th>
                  <th>Score</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {stats.recent_form.map((m, i) => (
                  <tr key={i}>
                    <td>{m.date}</td>
                    <td>{m.opponent}</td>
                    <td>{m.home_or_away}</td>
                    <td>{m.goals_for}–{m.goals_against}</td>
                    <td><FormBadge result={m.result} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section aria-label="Season statistics">
            <h3>Season Statistics</h3>
            <table className="stat-table full-width">
              <tbody>
                <tr><td>Matches Played</td><td>{stats.season_stats.matches_played}</td></tr>
                <tr><td>Wins</td><td>{stats.season_stats.wins}</td></tr>
                <tr><td>Draws</td><td>{stats.season_stats.draws}</td></tr>
                <tr><td>Losses</td><td>{stats.season_stats.losses}</td></tr>
                <tr><td>Goals Scored</td><td>{stats.season_stats.goals_scored}</td></tr>
                <tr><td>Goals Conceded</td><td>{stats.season_stats.goals_conceded}</td></tr>
                <tr>
                  <td>Win Rate</td>
                  <td>{Math.round(stats.season_stats.win_rate * 100)}%</td>
                </tr>
                <tr><td>ELO Rating</td><td>{stats.current_elo}</td></tr>
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  )
}
