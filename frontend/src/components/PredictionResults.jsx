import { useState } from 'react'
import TeamAnalysis from './TeamAnalysis'

function ProbBar({ label, prob, highlight }) {
  const pct = Math.round(prob * 100)
  return (
    <div className={`prob-row ${highlight ? 'prob-highlight' : ''}`}>
      <span className="prob-label">{label}</span>
      <div className="prob-bar-track" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
        <div className="prob-bar-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="prob-pct">{pct}%</span>
    </div>
  )
}

function FormBadge({ result }) {
  const cls = result === 'W' ? 'badge-win' : result === 'D' ? 'badge-draw' : 'badge-loss'
  return <span className={`form-badge ${cls}`}>{result}</span>
}

function TeamStats({ label, stats }) {
  if (!stats) return null
  return (
    <div className="team-stat-col">
      <h4>{label}</h4>
      <div className="form-badges">
        {(stats.form || []).map((r, i) => <FormBadge key={i} result={r} />)}
      </div>
      <table className="stat-table">
        <tbody>
          <tr><td>Goals Scored</td><td>{stats.goals_scored}</td></tr>
          <tr><td>Goals Conceded</td><td>{stats.goals_conceded}</td></tr>
          <tr><td>ELO Rating</td><td>{stats.elo}</td></tr>
        </tbody>
      </table>
    </div>
  )
}

function ExplanationPanel({ explanation }) {
  if (!explanation || explanation.length === 0) return null
  return (
    <section className="explanation-panel" aria-label="Why this prediction?">
      <h3>Why this prediction?</h3>
      <ol className="explanation-list">
        {explanation.map((item, i) => (
          <li key={i} className={`explanation-item ${item.direction}`}>
            <span className="expl-label">{item.label}</span>
            <span className="expl-badge">{item.direction === 'positive' ? '↑ favours home' : '↓ favours away'}</span>
          </li>
        ))}
      </ol>
      <p className="expl-note">
        Factors ordered by their SHAP contribution to the predicted outcome.
      </p>
    </section>
  )
}

export default function PredictionResults({ data, onReset }) {
  const [analysisTeam, setAnalysisTeam] = useState(null)
  const { home_team, away_team, prediction, home_stats, away_stats } = data
  const { probabilities, predicted_outcome, explanation } = prediction

  const isHomeWin = predicted_outcome === 'Home Win'
  const isDraw = predicted_outcome === 'Draw'
  const isAwayWin = predicted_outcome === 'Away Win'

  return (
    <div className="results-wrapper">
      <div className="card results-card">
        <button className="back-btn" onClick={onReset} aria-label="Back to fixture selection">
          ← New Prediction
        </button>

        <h2 className="fixture-title">{home_team} vs {away_team}</h2>
        <p className="predicted-outcome">Predicted: <strong>{predicted_outcome}</strong></p>

        <section className="prob-section" aria-label="Outcome probabilities">
          <ProbBar label="Home Win" prob={probabilities.home_win} highlight={isHomeWin} />
          <ProbBar label="Draw"     prob={probabilities.draw}     highlight={isDraw} />
          <ProbBar label="Away Win" prob={probabilities.away_win} highlight={isAwayWin} />
        </section>

        <section className="stats-section" aria-label="Team statistics">
          <h3>Team Statistics (Last 5 Matches)</h3>
          <div className="stats-cols">
            <TeamStats label={home_team} stats={home_stats} />
            <TeamStats label={away_team} stats={away_stats} />
          </div>
        </section>

        <ExplanationPanel explanation={explanation} />

        <div className="analysis-links">
          <button className="link-btn" onClick={() => setAnalysisTeam(home_team)}>
            View {home_team} Analysis
          </button>
          <button className="link-btn" onClick={() => setAnalysisTeam(away_team)}>
            View {away_team} Analysis
          </button>
        </div>

        <p className="disclaimer">
          Predictions are probabilistic estimates based on historical data.
          Football results are inherently uncertain.
        </p>
      </div>

      {analysisTeam && (
        <TeamAnalysis teamName={analysisTeam} onClose={() => setAnalysisTeam(null)} />
      )}
    </div>
  )
}
