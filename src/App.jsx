import { useMemo, useState } from 'react'
import { buildStandings, teamList } from './data/placeholder'

const gameweeks = Array.from({ length: 38 }, (_, index) => index + 1)

function sortByNumber(key) {
  return (a, b) => b[key] - a[key]
}

function TableCard({ title, rows, favoriteTeam, mode }) {
  return (
    <section className="card table-card">
      <header className="card-header">
        <div>
          <p className="eyebrow">{mode === 'current' ? 'Live Table' : 'Predicted Table'}</p>
          <h2>{title}</h2>
        </div>
        <span className="pill">{rows.length} clubs</span>
      </header>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Pos</th>
              <th>Team</th>
              <th>Pts</th>
              <th>Pred Pts</th>
              <th>?</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const delta = row.points - row.predictedPoints
              const deltaClass = delta > 0 ? 'delta up' : delta < 0 ? 'delta down' : 'delta flat'
              const isFavorite = row.team === favoriteTeam
              return (
                <tr key={`${mode}-${row.team}`} className={isFavorite ? 'favorite' : undefined}>
                  <td className="pos">{row.position}</td>
                  <td className="team">
                    <span className="team-badge" style={{ background: row.color }} />
                    <span>{row.team}</span>
                    {isFavorite && <span className="fav-tag">Favorite</span>}
                  </td>
                  <td>{row.points}</td>
                  <td>{row.predictedPoints}</td>
                  <td>
                    <span className={deltaClass}>{delta > 0 ? `+${delta}` : delta}</span>
                  </td>
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

export default function App() {
  const [gameweek, setGameweek] = useState(24)
  const [favoriteTeam, setFavoriteTeam] = useState(teamList[0].team)

  const { currentTable, predictedTable, topOver, topUnder } = useMemo(() => {
    const baseRows = buildStandings(gameweek)

    const currentTable = [...baseRows]
      .sort(sortByNumber('points'))
      .map((row, index) => ({ ...row, position: index + 1 }))

    const predictedTable = [...baseRows]
      .sort(sortByNumber('predictedPoints'))
      .map((row, index) => ({ ...row, position: index + 1 }))

    const deltas = baseRows
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
  }, [gameweek])

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
            Gameweek
            <select value={gameweek} onChange={(event) => setGameweek(Number(event.target.value))}>
              {gameweeks.map((week) => (
                <option key={week} value={week}>
                  GW {week}
                </option>
              ))}
            </select>
          </label>
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

      <section className="tables">
        <TableCard title="Current table" rows={currentTable} favoriteTeam={favoriteTeam} mode="current" />
        <TableCard title="Predicted table" rows={predictedTable} favoriteTeam={favoriteTeam} mode="predicted" />
      </section>
    </div>
  )
}
