import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [apiStatus, setApiStatus] = useState('checking')
  const [data, setData] = useState(null)

  useEffect(() => {
    checkApiHealth()
  }, [])

  const checkApiHealth = async () => {
    try {
      const response = await fetch('/api/health')
      if (response.ok) {
        const json = await response.json()
        setApiStatus('connected')
        setData(json)
      } else {
        setApiStatus('error')
      }
    } catch (error) {
      setApiStatus('error')
      console.error('API connection failed:', error)
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>CryptoVolt</h1>
        <p>AI-Powered Algorithmic Trading Platform</p>
      </header>

      <main className="main">
        <div className="status-card">
          <h2>API Status</h2>
          <div className={`status ${apiStatus}`}>
            {apiStatus === 'connected' && (
              <>
                <span className="status-indicator green"></span>
                <p>Connected to API</p>
                {data && <p className="status-detail">{data.message}</p>}
              </>
            )}
            {apiStatus === 'checking' && (
              <>
                <span className="status-indicator yellow"></span>
                <p>Checking API connection...</p>
              </>
            )}
            {apiStatus === 'error' && (
              <>
                <span className="status-indicator red"></span>
                <p>Failed to connect to API</p>
                <button onClick={checkApiHealth} className="retry-btn">
                  Retry
                </button>
              </>
            )}
          </div>
        </div>

        <div className="info-cards">
          <div className="info-card">
            <h3>Getting Started</h3>
            <ul>
              <li>Configure your trading strategies</li>
              <li>Set risk management parameters</li>
              <li>Monitor live signals and trades</li>
              <li>Review performance analytics</li>
            </ul>
          </div>

          <div className="info-card">
            <h3>Features</h3>
            <ul>
              <li>Hybrid Decision Engine (Rules + ML)</li>
              <li>Real-time Sentiment Analysis</li>
              <li>Paper Trading Simulation</li>
              <li>Advanced Risk Management</li>
            </ul>
          </div>

          <div className="info-card">
            <h3>Quick Links</h3>
            <ul>
              <li><a href="/api/docs">API Documentation</a></li>
              <li><a href="/strategies">Trading Strategies</a></li>
              <li><a href="/trades">Trade History</a></li>
              <li><a href="/analytics">Analytics</a></li>
            </ul>
          </div>
        </div>
      </main>

      <footer className="footer">
        <p>&copy; 2026 CryptoVolt - Research Grade Prototype</p>
      </footer>
    </div>
  )
}

export default App
