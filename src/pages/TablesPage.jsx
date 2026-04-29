import { SeasonSelector } from '../components/shared/SeasonSelector'
import { TableCard } from '../components/standings/TableCard'

export function TablesPage({
  season,
  seasonOptions,
  onSeasonChange,
  currentTable,
  predictedTable,
  favoriteTeam,
  gameweek,
  onGameweekChange,
}) {
  return (
    <div className="space-y-4">
      <section className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Standings</p>
          <h2 className="text-3xl font-semibold tracking-tight text-slate-900">Current vs Predicted Tables</h2>
          <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
            Compare live and projected positions by gameweek without crowding the rest of the dashboard.
          </p>
        </div>
        <SeasonSelector season={season} seasonOptions={seasonOptions} onSeasonChange={onSeasonChange} />
      </section>
      <section className="grid items-stretch gap-4 xl:grid-cols-2">
        <TableCard
          title="Current table"
          rows={currentTable}
          favoriteTeam={favoriteTeam}
          mode="current"
          gameweek={gameweek}
          onGameweekChange={onGameweekChange}
        />
        <TableCard
          title="Predicted table"
          rows={predictedTable}
          favoriteTeam={favoriteTeam}
          mode="predicted"
          gameweek={gameweek}
          onGameweekChange={onGameweekChange}
        />
      </section>
    </div>
  )
}
