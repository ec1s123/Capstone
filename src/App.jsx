import { useMemo, useState } from 'react'
import { buildStandings, teamList } from './data/placeholder'
import predictedTableData from './data/predicted_table.json'

const gameweeks = Array.from({ length: 38 }, (_, index) => index + 1)

function sortByNumber(key) {
  return (a, b) => b[key] - a[key]
}

function TableCard({ title, rows, favoriteTeam, mode, gameweek, onGameweekChange }) {
  const isCurrentMode = mode === 'current'
  const cardClassName = isCurrentMode ? 'table-card current-card' : 'table-card predicted-card'
  const sourceLabel = isCurrentMode ? 'Live Data' : 'Model Forecast'

  return (
    <section className={`card ${cardClassName}`}>
      <header className="card-header">
        <div>
          <p className="eyebrow">{mode === 'current' ? 'Live Table' : 'Predicted Table'}</p>
          <h2>{title}</h2>
        </div>
        <div className="header-meta">
          <label className="inline-gw-picker">
            GW
            <select value={gameweek} onChange={(event) => onGameweekChange(Number(event.target.value))}>
              {gameweeks.map((week) => (
                <option key={`${mode}-gw-${week}`} value={week}>
                  {week}
                </option>
              ))}
            </select>
          </label>
          <span className="source-chip">{sourceLabel}</span>
        </div>
      </header>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Pos</th>
              <th>Team</th>
              <th>P</th>
              <th>{isCurrentMode ? 'Pts' : 'Pred Pts'}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const isFavorite = row.team === favoriteTeam
              return (
                <tr key={`${mode}-${row.team}`} className={isFavorite ? 'favorite' : undefined}>
                  <td className="pos">{row.position}</td>
                  <td className="team">
                    <span className="team-badge" style={{ background: row.color }} />
                    <span>{row.team}</span>
                    {isFavorite && <span className="fav-tag">Favorite</span>}
                  </td>
                  <td>{row.played}</td>
                  <td>{isCurrentMode ? row.points : row.predictedPoints}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function PerformerList({ title, subtitle, items, favoriteTeam }) {
  return (
    <section className="card performer-card">
      <header className="card-header">
        <div>
          <p className="eyebrow">{subtitle}</p>
          <h3>{title}</h3>
        </div>
      </header>
      <div className="performer-list">
        {items.map((item) => {
          const isFavorite = item.team === favoriteTeam
          const deltaClass = item.delta > 0 ? 'delta up' : 'delta down'
          return (
            <div key={item.team} className={`performer ${isFavorite ? 'favorite' : ''}`}>
              <div>
                <p className="team-name">{item.team}</p>
                <p className="muted">Current {item.points} vs predicted {item.predictedPoints}</p>
              </div>
              <span className={deltaClass}>{item.delta > 0 ? `+${item.delta}` : item.delta}</span>
            </div>
          )
        })}
      </div>
    </section>
  )
}

function FinalModelTableCard({ rows, favoriteTeam, teamColors }) {
  return (
    <section className="card table-card">
      <header className="card-header">
        <div>
          <p className="eyebrow">Model Output</p>
          <h2>Predicted Final Premier League Table</h2>
        </div>
        <span className="pill">{rows.length} clubs</span>
      </header>
      {rows.length === 0 ? (
        <p className="muted">
          Run <code>.venv/bin/python src/MLMODEL.py</code> to generate{' '}
          <code>src/data/predicted_table.json</code>.
        </p>
      ) : (
        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>Pos</th>
                <th>Team</th>
                <th>P</th>
                <th>W</th>
                <th>D</th>
                <th>L</th>
                <th>Pts</th>
                <th>xPts</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const isFavorite = row.Team === favoriteTeam
                return (
                  <tr key={`model-${row.Team}`} className={isFavorite ? 'favorite' : undefined}>
                    <td className="pos">{row.Position}</td>
                    <td className="team">
                      <span className="team-badge" style={{ background: teamColors[row.Team] || '#8f9bb3' }} />
                      <span>{row.Team}</span>
                      {isFavorite && <span className="fav-tag">Favorite</span>}
                    </td>
                    <td>{row.Played}</td>
                    <td>{row.Won}</td>
                    <td>{row.Drawn}</td>
                    <td>{row.Lost}</td>
                    <td>{row.Points}</td>
                    <td>{row.ExpectedPoints}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

export default function App() {
  const [currentGameweek, setCurrentGameweek] = useState(24)
  const [predictedGameweek, setPredictedGameweek] = useState(24)
  const [favoriteTeam, setFavoriteTeam] = useState(teamList[0].team)
  const teamColors = useMemo(
    () => Object.fromEntries(teamList.map((team) => [team.team, team.color])),
    []
  )

  const { currentTable, predictedTable, topOver, topUnder } = useMemo(() => {
    const currentRows = buildStandings(currentGameweek)
    const predictedRows = buildStandings(predictedGameweek)

    const currentTable = [...currentRows]
      .sort(sortByNumber('points'))
      .map((row, index) => ({ ...row, position: index + 1 }))

    const predictedTable = [...predictedRows]
      .sort(sortByNumber('predictedPoints'))
      .map((row, index) => ({ ...row, position: index + 1 }))

    const deltas = currentRows
      .map((row) => ({
        team: row.team,
        points: row.points,
        predictedPoints: row.predictedPoints,
        delta: row.points - row.predictedPoints
      }))
      .sort((a, b) => b.delta - a.delta)

    const topOver = deltas.slice(0, 3)
    const topUnder = [...deltas].reverse().slice(0, 3)

    return { currentTable, predictedTable, topOver, topUnder }
  }, [currentGameweek, predictedGameweek])

  return (
    <div className="app">
      <div className="backdrop" />
      <header className="hero">
        <div>
          <p className="eyebrow">Premier League Predictor</p>
          <h1>See the table you have vs the table you think is coming.</h1>
          <p className="subhead">
            Compare current points to predicted points by gameweek, highlight your club, and surface the
            biggest over and under performers.
          </p>
        </div>
        <div className="hero-controls">
          <label>
            Favorite team
            <select value={favoriteTeam} onChange={(event) => setFavoriteTeam(event.target.value)}>
              {teamList.map((team) => (
                <option key={team.team} value={team.team}>
                  {team.team}
                </option>
              ))}
            </select>
          </label>
          <div className="note">Data is placeholder until CSV parsing is wired in.</div>
        </div>
      </header>

      <section className="insights">
        <PerformerList
          title="Biggest over-performers"
          subtitle="Points above predicted"
          items={topOver}
          favoriteTeam={favoriteTeam}
        />
        <PerformerList
          title="Biggest under-performers"
          subtitle="Points below predicted"
          items={topUnder}
          favoriteTeam={favoriteTeam}
        />
      </section>

      <section className="tables compare-layout">
        <TableCard
          title="Current table"
          rows={currentTable}
          favoriteTeam={favoriteTeam}
          mode="current"
          gameweek={currentGameweek}
          onGameweekChange={setCurrentGameweek}
        />
        <div className="compare-rail" aria-hidden="true">
          <span>NOW</span>
          <span className="rail-line" />
          <span>FORECAST</span>
        </div>
        <TableCard
          title="Predicted table"
          rows={predictedTable}
          favoriteTeam={favoriteTeam}
          mode="predicted"
          gameweek={predictedGameweek}
          onGameweekChange={setPredictedGameweek}
        />
      </section>

      <section className="tables single">
        <FinalModelTableCard
          rows={predictedTableData}
          favoriteTeam={favoriteTeam}
          teamColors={teamColors}
        />
      </section>
    </div>
  )
}
