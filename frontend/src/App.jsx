import { useState } from 'react'
import FixtureSelector from './components/FixtureSelector'
import PredictionResults from './components/PredictionResults'

export default function App() {
  const [predictionData, setPredictionData] = useState(null)

  return (
    <main className="app-shell">
      {predictionData === null ? (
        <FixtureSelector onPredict={(data) => setPredictionData(data)} />
      ) : (
        <PredictionResults data={predictionData} onReset={() => setPredictionData(null)} />
      )}
    </main>
  )
}
