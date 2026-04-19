import { useMemo, useState } from 'react'
import { Activity, BrainCircuit, Home, Sparkles, TrendingDown, TrendingUp } from 'lucide-react'

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
import { buildLastFiveForm, buildStandings, teamList } from './data/placeholder'
import predictedTableData from './data/predicted_table.json'
import { cn } from './lib/utils'

const gameweeks = Array.from({ length: 38 }, (_, index) => index + 1)
const methodology = [
  {
    title: 'Historical Performance',
    description: 'Long-term team patterns, matchup history, and season-level trends shape the baseline view.',
    icon: Activity,
  },
  {
    title: 'Recent Form Tracking',
    description: 'Short-term momentum, consistency, and changes in output are factored into every fixture forecast.',
    icon: TrendingUp,
  },
  {
    title: 'Match Context',
    description: 'Home and away strength, scoring profiles, and defensive stability refine the prediction for each game.',
    icon: Home,
  },
  {
    title: 'Model-Driven Probabilities',
    description: 'Structured model outputs turn raw football data into clearer, more actionable prediction signals.',
    icon: BrainCircuit,
  },
]

const teamAliases = {
  'Man City': 'Manchester City',
  'Man United': 'Manchester United',
  Newcastle: 'Newcastle United',
  "Nott'm Forest": 'Nottingham Forest',
}

const teamDomains = {
  Arsenal: 'arsenal.com',
  'Aston Villa': 'avfc.co.uk',
  Bournemouth: 'afcb.co.uk',
  Brentford: 'brentfordfc.com',
  Brighton: 'brightonandhovealbion.com',
  Burnley: 'burnleyfootballclub.com',
  Chelsea: 'chelseafc.com',
  'Crystal Palace': 'cpfc.co.uk',
  Everton: 'evertonfc.com',
  Fulham: 'fulhamfc.com',
  Ipswich: 'itfc.co.uk',
  Leicester: 'lcfc.com',
  Liverpool: 'liverpoolfc.com',
  'Luton Town': 'lutontown.co.uk',
  'Manchester City': 'mancity.com',
  'Manchester United': 'manutd.com',
  'Newcastle United': 'nufc.co.uk',
  'Nottingham Forest': 'nottinghamforest.co.uk',
  Southampton: 'southamptonfc.com',
  'Sheffield United': 'sufc.co.uk',
  Tottenham: 'tottenhamhotspur.com',
  'West Ham': 'whufc.com',
  Wolves: 'wolves.co.uk',
}

const toneMap = {
  current: {
    eyebrow: 'Live Table',
    source: 'Live Data',
    card: 'border-sky-200 bg-white',
    badge: 'border-sky-200 bg-sky-50 text-sky-700',
    header: 'bg-sky-50/80',
    favorite: 'bg-sky-50',
    pos: 'text-sky-700',
    accentBorder: 'border-sky-200',
  },
  predicted: {
    eyebrow: 'Predicted Table',
    source: 'Model Forecast',
    card: 'border-emerald-200 bg-white',
    badge: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    header: 'bg-emerald-50/80',
    favorite: 'bg-emerald-50',
    pos: 'text-emerald-700',
    accentBorder: 'border-emerald-200',
  },
}

function normalizeTeamName(team) {
  return teamAliases[team] ?? team
}

function getClubLogoUrl(team) {
  const normalizedTeam = normalizeTeamName(team)
  const domain = teamDomains[normalizedTeam]
  if (!domain) return null
  return `https://www.google.com/s2/favicons?domain=${domain}&sz=128`
}

function getTeamInitials(team) {
  return team
    .split(' ')
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase())
    .slice(0, 2)
    .join('')
}

function sortByNumber(key) {
  return (a, b) => b[key] - a[key]
}

function deltaClass(delta) {
  if (delta > 0) return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (delta < 0) return 'border-rose-200 bg-rose-50 text-rose-700'
  return 'border-slate-200 bg-slate-100 text-slate-600'
}

function formResultClass(result) {
  if (result === 'W') return 'bg-emerald-700 text-white'
  if (result === 'L') return 'bg-rose-600 text-white'
  if (result === 'D') return 'bg-slate-500 text-white'
  return 'bg-slate-200 text-slate-400'
}

function ClubLogo({ team }) {
  const [hasError, setHasError] = useState(false)
  const normalizedTeam = normalizeTeamName(team)
  const logoUrl = getClubLogoUrl(normalizedTeam)
  const initials = getTeamInitials(normalizedTeam)

  if (!logoUrl || hasError) {
    return (
      <span className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-slate-300 bg-white text-[9px] font-semibold text-slate-700">
        {initials}
      </span>
    )
  }

  return (
    <img
      src={logoUrl}
      alt={`${normalizedTeam} crest`}
      className="h-6 w-6 shrink-0 rounded-full border border-slate-200 bg-white object-cover p-[1px]"
      loading="lazy"
      onError={() => setHasError(true)}
    />
  )
}

function FormChips({ results }) {
  const safeResults = Array.isArray(results) ? results : [null, null, null, null, null]

  return (
    <div className="flex min-w-[150px] items-center gap-1">
      {safeResults.map((result, index) => (
        <span
          key={`${result ?? 'na'}-${index}`}
          className={cn(
            'inline-flex h-6 w-6 items-center justify-center rounded-sm text-[10px] font-bold tracking-[0.04em]',
            formResultClass(result)
          )}
          aria-label={result ? `Result ${result}` : 'No result'}
          title={result ?? 'No result'}
        >
          {result ?? '-'}
        </span>
      ))}
    </div>
  )
}

function TableCard({ title, rows, favoriteTeam, mode, gameweek, onGameweekChange }) {
  const tone = toneMap[mode]
  const isCurrentMode = mode === 'current'

  return (
    <Card
      className={cn(
        'overflow-hidden border shadow-sm',
        tone.card
      )}
    >
      <CardHeader className="pb-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-[0.2em] text-muted-foreground">{tone.eyebrow}</p>
            <CardTitle className="text-xl">{title}</CardTitle>
            <CardDescription className="mt-1 text-xs">
              Form (W = win, D = draw, L = loss) is shown as the last five matches, oldest to newest.
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Select value={String(gameweek)} onValueChange={(value) => onGameweekChange(Number(value))}>
              <SelectTrigger
                className={cn('h-9 w-[96px] bg-white text-xs text-slate-900', tone.accentBorder)}
              >
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
        <div className="overflow-hidden rounded-lg border border-slate-200">
          <Table>
            <TableHeader className={tone.header}>
              <TableRow className="border-b-0 hover:bg-transparent">
                <TableHead>Pos</TableHead>
                <TableHead>Team</TableHead>
                <TableHead className="text-right">P</TableHead>
                <TableHead className="text-right">{isCurrentMode ? 'Pts' : 'Pred Pts'}</TableHead>
                <TableHead className="min-w-[170px]">Form (Last 5)</TableHead>
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
                        <ClubLogo team={row.team} />
                        <span>{row.team}</span>
                        {isFavorite && (
                          <Badge
                            variant="outline"
                            className="ml-2 border-amber-200 bg-amber-50 text-[10px] uppercase tracking-[0.12em] text-amber-700"
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
                    <TableCell>
                      <FormChips results={isCurrentMode ? row.form : row.predictedForm} />
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
    <Card className="border-slate-200 bg-white shadow-sm">
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
                ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                : 'border-rose-200 bg-rose-50 text-rose-700'
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
                'flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 px-3 py-3',
                isFavorite && 'border-amber-200 bg-amber-50'
              )}
            >
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <ClubLogo team={item.team} />
                  <p className="text-sm font-semibold">{item.team}</p>
                </div>
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

function FinalModelTableCard({ rows, favoriteTeam }) {
  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Model Output</p>
            <CardTitle className="text-2xl">Predicted Final Premier League Table</CardTitle>
            <CardDescription className="mt-2">
              Softmax model projection loaded from <code>src/data/predicted_table.json</code>. Form uses W = win, D = draw, and L = loss over the latest five matches.
            </CardDescription>
          </div>
          <Badge variant="outline" className="gap-1 border-amber-200 bg-amber-50 text-amber-700">
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
          <div className="overflow-hidden rounded-lg border border-slate-200">
            <Table>
              <TableHeader className="bg-slate-50">
                <TableRow className="border-b-0 hover:bg-transparent">
                  <TableHead>Pos</TableHead>
                  <TableHead>Team</TableHead>
                  <TableHead className="text-right">P</TableHead>
                  <TableHead className="text-right">W</TableHead>
                  <TableHead className="text-right">D</TableHead>
                  <TableHead className="text-right">L</TableHead>
                  <TableHead className="text-right">Pts</TableHead>
                  <TableHead className="text-right">xPts</TableHead>
                  <TableHead className="min-w-[170px]">Form (Last 5)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => {
                  const normalizedRowTeam = normalizeTeamName(row.Team)
                  const isFavorite = normalizeTeamName(favoriteTeam) === normalizedRowTeam
                  const modelForm = buildLastFiveForm(normalizedRowTeam, Number(row.Played), 139)
                  return (
                    <TableRow
                      key={`model-${row.Team}`}
                      className={cn(isFavorite && 'bg-amber-50 hover:bg-amber-100/60')}
                    >
                      <TableCell className="w-14 font-semibold text-amber-700">{row.Position}</TableCell>
                      <TableCell className="font-medium">
                        <div className="flex items-center gap-2">
                          <ClubLogo team={normalizedRowTeam} />
                          <span>{row.Team}</span>
                          {isFavorite && (
                            <Badge
                              variant="outline"
                              className="ml-2 border-amber-200 bg-amber-50 text-[10px] uppercase tracking-[0.12em] text-amber-700"
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
                      <TableCell>
                        <FormChips results={modelForm} />
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
      <main className="relative z-10 mx-auto w-full max-w-[1320px] px-4 py-8 sm:px-6 md:py-12 lg:px-8 lg:py-16">
        <section className="grid items-start gap-6 lg:grid-cols-[1.35fr_0.9fr]">
          <div className="space-y-5">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Premier League Predictor</p>
            <h1 className="max-w-4xl text-4xl font-semibold leading-tight tracking-tight md:text-5xl">
              Smarter Premier League Predictions, Backed by Data, Not Guesswork
            </h1>
            <p className="max-w-3xl text-lg leading-relaxed text-slate-700">
              Match-by-match insights, team form analysis, and AI-driven predictions to help you stay ahead every
              gameweek.
            </p>
            <p className="max-w-3xl text-base leading-relaxed text-muted-foreground md:text-lg">
              This platform delivers data-driven predictions for every Premier League fixture. By combining
              statistical modelling, historical performance, and current team form, each forecast is designed to go
              beyond surface-level analysis and highlight the factors that genuinely influence match outcomes.
            </p>
            <div className="flex flex-wrap gap-2 pt-1">
              <Badge variant="outline" className="border-slate-200 bg-white text-slate-700">
                AI-driven forecasts
              </Badge>
              <Badge variant="outline" className="border-slate-200 bg-white text-slate-700">
                Match-by-match analysis
              </Badge>
              <Badge variant="outline" className="border-slate-200 bg-white text-slate-700">
                Structured model insights
              </Badge>
            </div>
          </div>
          <Card className="border-slate-200 bg-white shadow-sm lg:ml-auto lg:w-full lg:max-w-md">
            <CardHeader className="pb-4">
              <CardTitle className="text-lg">Prediction Snapshot</CardTitle>
              <CardDescription>Track one club and compare current output against the projected model view.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-0">
              <div className="space-y-2">
                <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Favorite Club</p>
                <Select value={favoriteTeam} onValueChange={setFavoriteTeam}>
                  <SelectTrigger className="border-slate-300 bg-white text-slate-900">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {teamList.map((team) => (
                      <SelectItem key={team.team} value={team.team}>
                        <div className="flex items-center gap-2">
                          <ClubLogo team={team.team} />
                          <span>{team.team}</span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {favoriteSnapshot && (
                <div className="grid grid-cols-3 gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3">
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
                Predictions on this page are driven by the current project dataset and model output available in the
                app.
              </p>
            </CardContent>
          </Card>
        </section>

        <section className="mt-8">
          <Card className="border-slate-200 bg-white shadow-sm">
            <CardHeader className="pb-4">
              <p className="text-xs uppercase tracking-[0.2em] text-slate-500">How It Works</p>
              <CardTitle className="text-2xl">Every prediction is built on a structured approach.</CardTitle>
              <CardDescription className="max-w-3xl">
                Rather than relying on guesswork, the platform focuses on measurable signals that shape results across
                a full Premier League season.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              {methodology.map((item) => {
                const Icon = item.icon
                return (
                  <div key={item.title} className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                    <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-slate-200">
                      <Icon className="h-5 w-5 text-slate-700" />
                    </div>
                    <h3 className="text-base font-semibold text-slate-900">{item.title}</h3>
                    <p className="mt-2 text-sm leading-6 text-slate-600">{item.description}</p>
                  </div>
                )
              })}
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
              <div className="h-16 w-[2px] rounded-full bg-slate-300" />
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
          />
        </section>
      </main>
    </div>
  )
}
