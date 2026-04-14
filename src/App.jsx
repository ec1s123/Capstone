import { useMemo, useState } from 'react'
import { Sparkles, TrendingDown, TrendingUp } from 'lucide-react'

import { Badge } from './components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from './components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from './components/ui/table'
import { buildStandings, teamList } from './data/placeholder'
import predictedTableData from './data/predicted_table.json'
import { cn } from './lib/utils'

const gameweeks = Array.from({ length: 38 }, (_, index) => index + 1)

const teamAliases = {
  'Man City': 'Manchester City',
  'Man United': 'Manchester United',
  Newcastle: 'Newcastle United',
  "Nott'm Forest": 'Nottingham Forest',
}

const toneMap = {
  current: {
    eyebrow: 'Live Table',
    source: 'Live Data',
    card: 'border-sky-400/40 bg-slate-950/55',
    badge: 'border-sky-300/35 bg-sky-400/15 text-sky-200',
    header: 'bg-sky-500/10',
    favorite: 'bg-sky-500/12',
    pos: 'text-sky-300',
    accentBorder: 'border-sky-300/35',
  },
  predicted: {
    eyebrow: 'Predicted Table',
    source: 'Model Forecast',
    card: 'border-emerald-400/40 bg-slate-950/55',
    badge: 'border-emerald-300/35 bg-emerald-400/15 text-emerald-200',
    header: 'bg-emerald-500/10',
    favorite: 'bg-emerald-500/12',
    pos: 'text-emerald-300',
    accentBorder: 'border-emerald-300/35',
  },
}

function normalizeTeamName(team) {
  return teamAliases[team] ?? team
}

function sortByNumber(key) {
  return (a, b) => b[key] - a[key]
}

function deltaClass(delta) {
  if (delta > 0) return 'border-emerald-300/35 bg-emerald-400/20 text-emerald-100'
  if (delta < 0) return 'border-rose-300/35 bg-rose-500/20 text-rose-100'
  return 'border-white/20 bg-white/10 text-muted-foreground'
}

function TableCard({ title, rows, favoriteTeam, mode, gameweek, onGameweekChange }) {
  const tone = toneMap[mode]
  const isCurrentMode = mode === 'current'

  return (
    <Card
      className={cn(
        'overflow-hidden border backdrop-blur-xl shadow-[0_26px_70px_rgba(8,15,35,0.42)]',
        tone.card
      )}
    >
      <CardHeader className="pb-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-[0.2em] text-muted-foreground">{tone.eyebrow}</p>
            <CardTitle className="text-xl">{title}</CardTitle>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Select value={String(gameweek)} onValueChange={(value) => onGameweekChange(Number(value))}>
              <SelectTrigger className={cn('h-9 w-[96px] bg-black/20 text-xs', tone.accentBorder)}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {gameweeks.map((week) => (
                  <SelectItem key={`${mode}-gw-${week}`} value={String(week)}>
                    GW {week}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Badge variant="outline" className={cn('uppercase tracking-[0.14em]', tone.badge)}>
              {tone.source}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="overflow-hidden rounded-lg border border-white/10">
          <Table>
            <TableHeader className={tone.header}>
              <TableRow className="border-b-0 hover:bg-transparent">
                <TableHead>Pos</TableHead>
                <TableHead>Team</TableHead>
                <TableHead className="text-right">P</TableHead>
                <TableHead className="text-right">{isCurrentMode ? 'Pts' : 'Pred Pts'}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => {
                const isFavorite = row.team === favoriteTeam
                return (
                  <TableRow key={`${mode}-${row.team}`} className={cn(isFavorite && tone.favorite)}>
                    <TableCell className={cn('w-14 font-semibold', tone.pos)}>{row.position}</TableCell>
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        <span
                          className="h-2.5 w-2.5 rounded-full border border-white/20"
                          style={{ backgroundColor: row.color }}
                        />
                        <span>{row.team}</span>
                        {isFavorite && (
                          <Badge
                            variant="outline"
                            className="ml-2 border-amber-300/35 bg-amber-500/15 text-[10px] uppercase tracking-[0.12em] text-amber-100"
                          >
                            Favorite
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{row.played}</TableCell>
                    <TableCell className="text-right font-semibold tabular-nums">
                      {isCurrentMode ? row.points : row.predictedPoints}
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}

function PerformerList({ title, subtitle, items, favoriteTeam, direction }) {
  const isUp = direction === 'up'
  const Icon = isUp ? TrendingUp : TrendingDown

  return (
    <Card className="border-white/15 bg-slate-950/55 backdrop-blur-xl shadow-[0_22px_65px_rgba(8,15,35,0.4)]">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-[0.2em] text-muted-foreground">{subtitle}</p>
            <CardTitle className="text-xl">{title}</CardTitle>
          </div>
          <Badge
            variant="outline"
            className={cn(
              'gap-1 border text-[11px] uppercase tracking-[0.14em]',
              isUp
                ? 'border-emerald-300/35 bg-emerald-400/15 text-emerald-100'
                : 'border-rose-300/35 bg-rose-500/15 text-rose-100'
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            {isUp ? 'Over' : 'Under'}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        {items.map((item) => {
          const isFavorite = item.team === favoriteTeam
          return (
            <div
              key={item.team}
              className={cn(
                'flex items-center justify-between rounded-lg border border-white/10 bg-white/5 px-3 py-3',
                isFavorite && 'border-amber-300/35 bg-amber-500/10'
              )}
            >
              <div className="space-y-0.5">
                <p className="text-sm font-semibold">{item.team}</p>
                <p className="text-xs text-muted-foreground">
                  Current {item.points} vs predicted {item.predictedPoints}
                </p>
              </div>
              <span
                className={cn(
                  'rounded-full border px-2.5 py-1 text-xs font-semibold tabular-nums',
                  deltaClass(item.delta)
                )}
              >
                {item.delta > 0 ? `+${item.delta}` : item.delta}
              </span>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

function FinalModelTableCard({ rows, favoriteTeam, teamColors }) {
  return (
    <Card className="border-white/15 bg-slate-950/55 backdrop-blur-xl shadow-[0_30px_80px_rgba(7,13,31,0.45)]">
      <CardHeader className="pb-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Model Output</p>
            <CardTitle className="text-2xl">Predicted Final Premier League Table</CardTitle>
            <CardDescription className="mt-2">
              Softmax model projection loaded from <code>src/data/predicted_table.json</code>.
            </CardDescription>
          </div>
          <Badge variant="outline" className="gap-1 border-amber-300/35 bg-amber-400/15 text-amber-100">
            <Sparkles className="h-3.5 w-3.5" />
            {rows.length} clubs
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Run <code>python src/MLMODEL.py</code> to regenerate <code>src/data/predicted_table.json</code>.
          </p>
        ) : (
          <div className="overflow-hidden rounded-lg border border-white/10">
            <Table>
              <TableHeader className="bg-white/5">
                <TableRow className="border-b-0 hover:bg-transparent">
                  <TableHead>Pos</TableHead>
                  <TableHead>Team</TableHead>
                  <TableHead className="text-right">P</TableHead>
                  <TableHead className="text-right">W</TableHead>
                  <TableHead className="text-right">D</TableHead>
                  <TableHead className="text-right">L</TableHead>
                  <TableHead className="text-right">Pts</TableHead>
                  <TableHead className="text-right">xPts</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => {
                  const normalizedRowTeam = normalizeTeamName(row.Team)
                  const isFavorite = normalizeTeamName(favoriteTeam) === normalizedRowTeam
                  return (
                    <TableRow
                      key={`model-${row.Team}`}
                      className={cn(isFavorite && 'bg-amber-500/10 hover:bg-amber-500/15')}
                    >
                      <TableCell className="w-14 font-semibold text-amber-200">{row.Position}</TableCell>
                      <TableCell className="font-medium">
                        <div className="flex items-center gap-2">
                          <span
                            className="h-2.5 w-2.5 rounded-full border border-white/20"
                            style={{ backgroundColor: teamColors[normalizedRowTeam] || '#8ea2c0' }}
                          />
                          <span>{row.Team}</span>
                          {isFavorite && (
                            <Badge
                              variant="outline"
                              className="ml-2 border-amber-300/35 bg-amber-500/15 text-[10px] uppercase tracking-[0.12em] text-amber-100"
                            >
                              Favorite
                            </Badge>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{row.Played}</TableCell>
                      <TableCell className="text-right tabular-nums">{row.Won}</TableCell>
                      <TableCell className="text-right tabular-nums">{row.Drawn}</TableCell>
                      <TableCell className="text-right tabular-nums">{row.Lost}</TableCell>
                      <TableCell className="text-right font-semibold tabular-nums">{row.Points}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        {Number(row.ExpectedPoints).toFixed(2)}
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default function App() {
  const [currentGameweek, setCurrentGameweek] = useState(24)
  const [predictedGameweek, setPredictedGameweek] = useState(24)
  const [favoriteTeam, setFavoriteTeam] = useState(teamList[0].team)

  const teamColors = useMemo(
    () => Object.fromEntries(teamList.map((team) => [normalizeTeamName(team.team), team.color])),
    []
  )

  const { currentTable, predictedTable, topOver, topUnder, favoriteSnapshot } = useMemo(() => {
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
        delta: row.points - row.predictedPoints,
      }))
      .sort((a, b) => b.delta - a.delta)

    const favoriteCurrent = currentRows.find((row) => row.team === favoriteTeam)
    const favoritePredicted = predictedRows.find((row) => row.team === favoriteTeam)
    const favoriteSnapshot =
      favoriteCurrent && favoritePredicted
        ? {
            points: favoriteCurrent.points,
            predictedPoints: favoritePredicted.predictedPoints,
            delta: favoriteCurrent.points - favoritePredicted.predictedPoints,
          }
        : null

    return {
      currentTable,
      predictedTable,
      topOver: deltas.slice(0, 3),
      topUnder: [...deltas].reverse().slice(0, 3),
      favoriteSnapshot,
    }
  }, [currentGameweek, favoriteTeam, predictedGameweek])

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <div className="pointer-events-none absolute inset-0 -z-10 bg-[radial-gradient(circle_at_20%_8%,rgba(14,116,144,0.24),transparent_44%),radial-gradient(circle_at_82%_15%,rgba(16,185,129,0.2),transparent_40%),radial-gradient(circle_at_50%_94%,rgba(250,204,21,0.14),transparent_48%)]" />
      <main className="relative z-10 mx-auto w-full max-w-[1320px] px-4 py-8 sm:px-6 md:py-12 lg:px-8 lg:py-16">
        <section className="grid items-start gap-6 lg:grid-cols-[1.35fr_0.9fr]">
          <div className="space-y-4">
            <p className="text-xs uppercase tracking-[0.2em] text-sky-200/80">Premier League Predictor</p>
            <h1 className="max-w-3xl text-4xl font-semibold leading-tight tracking-tight md:text-5xl">
              Compare where the table is now against where your model thinks it will finish.
            </h1>
            <p className="max-w-2xl text-base leading-relaxed text-muted-foreground md:text-lg">
              Shadcn components now drive the UI for cleaner hierarchy, tighter spacing, and stronger visual
              consistency across controls, insights, and model output.
            </p>
          </div>
          <Card className="border-white/15 bg-slate-950/55 backdrop-blur-xl shadow-[0_24px_70px_rgba(8,15,35,0.45)] lg:ml-auto lg:w-full lg:max-w-md">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg">Session Controls</CardTitle>
              <CardDescription>Pin a club and track current vs projected output instantly.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-0">
              <div className="space-y-2">
                <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Favorite Club</p>
                <Select value={favoriteTeam} onValueChange={setFavoriteTeam}>
                  <SelectTrigger className="border-white/20 bg-black/20">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {teamList.map((team) => (
                      <SelectItem key={team.team} value={team.team}>
                        {team.team}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {favoriteSnapshot && (
                <div className="grid grid-cols-3 gap-2 rounded-lg border border-white/10 bg-white/5 p-3">
                  <div>
                    <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Current</p>
                    <p className="mt-1 text-sm font-semibold tabular-nums">{favoriteSnapshot.points} pts</p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Predicted</p>
                    <p className="mt-1 text-sm font-semibold tabular-nums">
                      {favoriteSnapshot.predictedPoints} pts
                    </p>
                  </div>
                  <div>
                    <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Delta</p>
                    <p className="mt-1 text-sm font-semibold tabular-nums">
                      {favoriteSnapshot.delta > 0 ? '+' : ''}
                      {favoriteSnapshot.delta}
                    </p>
                  </div>
                </div>
              )}
              <p className="text-xs leading-relaxed text-muted-foreground">
                Placeholder table simulation remains active until live CSV ingestion is wired to the frontend.
              </p>
            </CardContent>
          </Card>
        </section>

        <section className="mt-6 grid gap-6 md:grid-cols-2">
          <PerformerList
            title="Biggest over-performers"
            subtitle="Points above predicted"
            items={topOver}
            favoriteTeam={favoriteTeam}
            direction="up"
          />
          <PerformerList
            title="Biggest under-performers"
            subtitle="Points below predicted"
            items={topUnder}
            favoriteTeam={favoriteTeam}
            direction="down"
          />
        </section>

        <section className="mt-6 grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_72px_minmax(0,1fr)]">
          <TableCard
            title="Current table"
            rows={currentTable}
            favoriteTeam={favoriteTeam}
            mode="current"
            gameweek={currentGameweek}
            onGameweekChange={setCurrentGameweek}
          />
          <div className="hidden h-full min-h-[110px] items-center justify-center xl:flex">
            <div className="flex flex-col items-center gap-3 text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
              <span>Now</span>
              <div className="h-16 w-[2px] rounded-full bg-gradient-to-b from-sky-400 via-cyan-300 to-emerald-400" />
              <span>Forecast</span>
            </div>
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

        <section className="mt-6">
          <FinalModelTableCard
            rows={predictedTableData}
            favoriteTeam={favoriteTeam}
            teamColors={teamColors}
          />
        </section>
      </main>
    </div>
  )
}
