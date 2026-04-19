import { useMemo, useState } from 'react'
import { Activity, BrainCircuit, Home, Sparkles, TrendingDown, TrendingUp } from 'lucide-react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'

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
import premOddsPredictionsRaw from './data/prem_odds_predictions_2024_25.csv?raw'
import { cn } from './lib/utils'

const gameweeks = Array.from({ length: 38 }, (_, index) => index + 1)

const navItems = [
  { to: '/', label: 'Overview' },
  { to: '/tables', label: 'Tables' },
  { to: '/club', label: 'Club' },
  { to: '/methodology', label: 'Methodology' },
  { to: '/model-output', label: 'Model Output' },
]

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
  'Wolverhampton Wanderers': 'wolves.co.uk',
}

const toneMap = {
  current: {
    source: 'Live Data',
    card: 'border-sky-200 bg-white',
    badge: 'border-sky-200 bg-sky-50 text-sky-700',
    header: 'bg-sky-50/80',
    favorite: 'bg-sky-50 hover:bg-sky-100/70',
    pos: 'text-sky-700',
    accentBorder: 'border-sky-200',
  },
  predicted: {
    source: 'Model Forecast',
    card: 'border-emerald-200 bg-white',
    badge: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    header: 'bg-emerald-50/80',
    favorite: 'bg-emerald-50 hover:bg-emerald-100/70',
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

const outcomeLabelMap = {
  W: 'Win',
  D: 'Draw',
  L: 'Loss',
}

function parsePredictionFixtures(rawCsv) {
  const rows = rawCsv.trim().split(/\r?\n/)
  if (rows.length < 2) return []

  const headers = rows[0].split(',')
  const headerIndex = Object.fromEntries(headers.map((header, index) => [header, index]))
  const readValue = (values, key) => values[headerIndex[key]] ?? ''
  const asNumber = (value) => {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : 0
  }

  return rows
    .slice(1)
    .filter(Boolean)
    .map((line) => {
      const values = line.split(',')
      const homeTeam = normalizeTeamName(readValue(values, 'HomeTeam'))
      const awayTeam = normalizeTeamName(readValue(values, 'AwayTeam'))
      return {
        id: `${readValue(values, 'MatchDate')}-${homeTeam}-${awayTeam}`,
        season: readValue(values, 'Season'),
        matchDate: readValue(values, 'MatchDate'),
        homeTeam,
        awayTeam,
        fullTimeResult: readValue(values, 'FTR'),
        modelHomeProb: asNumber(readValue(values, 'ModelHomeProb')),
        modelDrawProb: asNumber(readValue(values, 'ModelDrawProb')),
        modelAwayProb: asNumber(readValue(values, 'ModelAwayProb')),
      }
    })
}

function deriveModelPickCode(match) {
  const probabilities = [
    ['H', match.modelHomeProb],
    ['D', match.modelDrawProb],
    ['A', match.modelAwayProb],
  ]

  let topPick = probabilities[0]
  for (const probabilityEntry of probabilities.slice(1)) {
    if (probabilityEntry[1] > topPick[1]) {
      topPick = probabilityEntry
    }
  }
  return topPick[0]
}

function outcomeForClub(resultCode, isHome) {
  if (resultCode === 'D') return 'D'
  const isClubWin = (resultCode === 'H' && isHome) || (resultCode === 'A' && !isHome)
  return isClubWin ? 'W' : 'L'
}

function outcomeBadgeClass(outcome) {
  if (outcome === 'W') return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (outcome === 'L') return 'border-rose-200 bg-rose-50 text-rose-700'
  return 'border-slate-200 bg-slate-100 text-slate-700'
}

function confidenceBadgeClass(value) {
  if (value >= 0.6) return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (value >= 0.45) return 'border-amber-200 bg-amber-50 text-amber-700'
  return 'border-slate-200 bg-slate-100 text-slate-700'
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function formatSigned(value, decimals = 1) {
  const fixed = value.toFixed(decimals)
  return value > 0 ? `+${fixed}` : fixed
}

function projectExpectedRecord(row) {
  const played = row.Played
  let bestRecord = { won: 0, drawn: 0, lost: played, points: 0 }
  let bestKey = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, 0, 0]

  for (let won = 0; won <= played; won += 1) {
    for (let drawn = 0; drawn <= played - won; drawn += 1) {
      const lost = played - won - drawn
      const points = won * 3 + drawn
      const pointsGap = Math.abs(points - row.ExpectedPoints)
      const shapeGap =
        Math.abs(won - row.expectedWins) +
        Math.abs(drawn - row.expectedDraws) +
        Math.abs(lost - row.expectedLosses)
      const score = pointsGap * 4 + shapeGap
      const key = [score, pointsGap, shapeGap, -won, -drawn]

      let isBetter = false
      for (let index = 0; index < key.length; index += 1) {
        if (key[index] < bestKey[index]) {
          isBetter = true
          break
        }
        if (key[index] > bestKey[index]) break
      }

      if (isBetter) {
        bestKey = key
        bestRecord = { won, drawn, lost, points }
      }
    }
  }

  return bestRecord
}

function buildModelOutputTable(fixtures) {
  const tableByTeam = new Map()

  const ensureTeam = (team) => {
    if (!tableByTeam.has(team)) {
      tableByTeam.set(team, {
        Team: team,
        Played: 0,
        Won: 0,
        Drawn: 0,
        Lost: 0,
        Points: 0,
        ExpectedPoints: 0,
        expectedWins: 0,
        expectedDraws: 0,
        expectedLosses: 0,
        formResults: [],
      })
    }
    return tableByTeam.get(team)
  }

  const orderedFixtures = [...fixtures].sort(
    (a, b) =>
      a.matchDate.localeCompare(b.matchDate) ||
      a.homeTeam.localeCompare(b.homeTeam) ||
      a.awayTeam.localeCompare(b.awayTeam)
  )

  orderedFixtures.forEach((fixture) => {
    const homeRow = ensureTeam(fixture.homeTeam)
    const awayRow = ensureTeam(fixture.awayTeam)

    homeRow.Played += 1
    awayRow.Played += 1

    homeRow.expectedWins += fixture.modelHomeProb
    homeRow.expectedDraws += fixture.modelDrawProb
    homeRow.expectedLosses += fixture.modelAwayProb
    homeRow.ExpectedPoints += fixture.modelHomeProb * 3 + fixture.modelDrawProb

    awayRow.expectedWins += fixture.modelAwayProb
    awayRow.expectedDraws += fixture.modelDrawProb
    awayRow.expectedLosses += fixture.modelHomeProb
    awayRow.ExpectedPoints += fixture.modelAwayProb * 3 + fixture.modelDrawProb

    const modelPick = deriveModelPickCode(fixture)
    homeRow.formResults.push(outcomeForClub(modelPick, true))
    awayRow.formResults.push(outcomeForClub(modelPick, false))
  })

  return [...tableByTeam.values()]
    .map((row) => {
      const projected = projectExpectedRecord(row)
      const form = row.formResults.slice(-5)
      while (form.length < 5) form.unshift(null)

      return {
        Team: row.Team,
        Played: row.Played,
        Won: projected.won,
        Drawn: projected.drawn,
        Lost: projected.lost,
        Points: projected.points,
        ExpectedPoints: Number(row.ExpectedPoints.toFixed(2)),
        Form: form,
      }
    })
    .sort((a, b) => b.ExpectedPoints - a.ExpectedPoints || b.Points - a.Points || a.Team.localeCompare(b.Team))
    .map((row, index) => ({ ...row, Position: index + 1 }))
}

function comparisonDeltaClass(delta, inverse = false) {
  const adjusted = inverse ? -delta : delta
  if (adjusted > 0.05) return 'text-emerald-700'
  if (adjusted < -0.05) return 'text-rose-700'
  return 'text-slate-600'
}

const logoSizeMap = {
  sm: {
    fallback: 'h-5 w-5 text-[8px]',
    image: 'h-5 w-5',
  },
  md: {
    fallback: 'h-6 w-6 text-[9px]',
    image: 'h-6 w-6',
  },
  lg: {
    fallback: 'h-10 w-10 text-xs',
    image: 'h-10 w-10',
  },
  xl: {
    fallback: 'h-14 w-14 text-sm',
    image: 'h-14 w-14',
  },
}

function ClubLogo({ team, size = 'md' }) {
  const [hasError, setHasError] = useState(false)
  const normalizedTeam = normalizeTeamName(team)
  const logoUrl = getClubLogoUrl(normalizedTeam)
  const initials = getTeamInitials(normalizedTeam)
  const sizeClass = logoSizeMap[size] ?? logoSizeMap.md

  if (!logoUrl || hasError) {
    return (
      <span
        className={cn(
          'inline-flex shrink-0 items-center justify-center rounded-full border border-slate-300 bg-white font-semibold text-slate-700',
          sizeClass.fallback
        )}
      >
        {initials}
      </span>
    )
  }

  return (
    <img
      src={logoUrl}
      alt={`${normalizedTeam} crest`}
      className={cn(
        'shrink-0 rounded-full border border-slate-200 bg-white object-cover p-[1px]',
        sizeClass.image
      )}
      loading="lazy"
      onError={() => setHasError(true)}
    />
  )
}

function FormChips({ results }) {
  const safeResults = Array.isArray(results) ? results : [null, null, null, null, null]

  return (
    <div className="flex items-center gap-1 whitespace-nowrap">
      {safeResults.map((result, index) => (
        <span
          key={`${result ?? 'na'}-${index}`}
          className={cn(
            'inline-flex h-5 w-5 items-center justify-center rounded-sm text-[9px] font-bold tracking-[0.04em]',
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
  const showDeltaColumn = mode === 'predicted'
  const teamColumnClass = showDeltaColumn ? 'w-[180px]' : 'w-[220px]'
  const formColumnClass = showDeltaColumn ? 'w-[132px]' : 'w-[148px]'

  return (
    <Card className={cn('h-full overflow-hidden border shadow-sm', tone.card)}>
      <CardHeader className="px-4 pb-4 pt-4 sm:px-5">
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
      <CardContent className="px-4 pb-4 pt-0 sm:px-5">
        <div className="overflow-x-auto overflow-y-hidden rounded-lg border border-slate-200">
          <Table className="table-fixed min-w-full">
            <TableHeader className={tone.header}>
              <TableRow className="border-b-0 hover:bg-transparent">
                <TableHead className="w-[52px] whitespace-nowrap">Pos</TableHead>
                <TableHead className={cn(teamColumnClass, 'whitespace-nowrap')}>Team</TableHead>
                <TableHead className="w-[48px] whitespace-nowrap text-right">P</TableHead>
                <TableHead className="w-[62px] whitespace-nowrap text-right">{isCurrentMode ? 'Pts' : 'Pred'}</TableHead>
                {showDeltaColumn && <TableHead className="w-[88px] whitespace-nowrap text-right">+/- vs Real</TableHead>}
                <TableHead className={cn(formColumnClass, 'whitespace-nowrap')}>Form (Last 5)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => {
                const isFavorite = row.team === favoriteTeam
                return (
                  <TableRow key={`${mode}-${row.team}`} className={cn('h-14', isFavorite && tone.favorite)}>
                    <TableCell className={cn('w-14 font-semibold', tone.pos)}>{row.position}</TableCell>
                    <TableCell className={cn(teamColumnClass, 'font-medium')}>
                      <div className="flex items-center gap-2">
                        <ClubLogo team={row.team} />
                        <span className="whitespace-nowrap">{row.team}</span>
                      </div>
                    </TableCell>
                    <TableCell className="w-[48px] text-right tabular-nums">{row.played}</TableCell>
                    <TableCell className="w-[62px] text-right font-semibold tabular-nums">
                      {isCurrentMode ? row.points : row.predictedPoints}
                    </TableCell>
                    {showDeltaColumn && (
                      <TableCell className="w-[88px] text-right">
                        <span
                          className={cn(
                            'inline-flex min-w-[54px] justify-center rounded-full border px-2.5 py-1 text-xs font-semibold tabular-nums',
                            deltaClass(row.delta)
                          )}
                        >
                          {row.delta > 0 ? `+${row.delta}` : row.delta}
                        </span>
                      </TableCell>
                    )}
                    <TableCell className={formColumnClass}>
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
              Softmax model projection derived from <code>src/data/prem_odds_predictions_2024_25.csv</code>. Form
              uses W = win, D = draw, and L = loss over the latest five predicted matches. Pts/W/D/L are an
              expectation-fit projection; xPts comes directly from match probabilities.
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
            Run <code>python src/MLMODEL.py</code> to regenerate <code>src/data/prem_odds_predictions_2024_25.csv</code>.
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
                        <FormChips results={row.Form} />
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

function FavoriteTeamCard({ favoriteTeam, onFavoriteTeamChange, favoriteSnapshot }) {
  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-4">
        <CardTitle className="text-lg">Prediction Snapshot</CardTitle>
        <CardDescription>Track one club and compare current output against the projected model view.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 pt-0">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Favorite Club</p>
          <Select value={favoriteTeam} onValueChange={onFavoriteTeamChange}>
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
              <p className="mt-1 text-sm font-semibold tabular-nums">{favoriteSnapshot.predictedPoints} pts</p>
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
      </CardContent>
    </Card>
  )
}

function OverviewPage({ topOver, topUnder, favoriteTeam, favoriteSnapshot, onFavoriteTeamChange }) {
  return (
    <div className="space-y-6">
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
            This platform delivers data-driven predictions for every Premier League fixture. By combining statistical
            modelling, historical performance, and current team form, each forecast is designed to go beyond
            surface-level analysis and highlight the factors that genuinely influence match outcomes.
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
        <FavoriteTeamCard
          favoriteTeam={favoriteTeam}
          onFavoriteTeamChange={onFavoriteTeamChange}
          favoriteSnapshot={favoriteSnapshot}
        />
      </section>

      <section className="grid gap-6 md:grid-cols-2">
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
    </div>
  )
}

function TablesPage({
  currentTable,
  predictedTable,
  favoriteTeam,
  gameweek,
  onGameweekChange,
}) {
  return (
    <div className="space-y-4">
      <section className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Standings</p>
        <h2 className="text-3xl font-semibold tracking-tight text-slate-900">Current vs Predicted Tables</h2>
        <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
          Compare live and projected positions by gameweek without crowding the rest of the dashboard.
        </p>
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

function MethodologyPage() {
  return (
    <section className="space-y-4">
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">How It Works</p>
        <h2 className="text-3xl font-semibold tracking-tight text-slate-900">
          Every prediction is built on a structured approach.
        </h2>
        <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
          Rather than relying on guesswork, the platform focuses on measurable signals that shape results across a
          full Premier League season.
        </p>
      </div>
      <Card className="border-slate-200 bg-white shadow-sm">
        <CardContent className="grid gap-4 p-6 md:grid-cols-2 xl:grid-cols-4">
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
  )
}

function ModelOutputPage({ favoriteTeam, modelOutputTable }) {
  return (
    <section className="space-y-4">
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Model Output</p>
        <h2 className="text-3xl font-semibold tracking-tight text-slate-900">Predicted Final Table</h2>
        <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
          View the complete softmax projection in a dedicated page without the rest of the dashboard content.
        </p>
      </div>
      <FinalModelTableCard rows={modelOutputTable} favoriteTeam={favoriteTeam} />
    </section>
  )
}

function ClubPage({ clubs, selectedClub, onSelectedClubChange, clubFixtures, clubSummary }) {
  return (
    <section className="space-y-4">
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Club View</p>
        <h2 className="text-3xl font-semibold tracking-tight text-slate-900">Fixtures, Results, and Model Confidence</h2>
        <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
          Select any club to view all fixtures, match outcomes, and the model confidence on each prediction.
        </p>
      </div>

      <Card className="border-slate-200 bg-white shadow-sm">
        <CardHeader className="space-y-5 pb-4">
          <div>
            <CardTitle className="text-lg">Club Breakdown</CardTitle>
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 sm:p-5">
            <p className="mb-2 text-xs uppercase tracking-[0.16em] text-muted-foreground">Choose Club</p>
            <Select value={selectedClub} onValueChange={onSelectedClubChange}>
              <SelectTrigger className="h-16 w-full border-2 border-slate-300 bg-white px-5 text-lg font-semibold text-slate-900 shadow-sm sm:h-20 sm:text-2xl">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-[420px]">
                {clubs.map((club) => (
                  <SelectItem key={club} value={club} className="py-3 text-lg">
                    <div className="flex items-center gap-3">
                      <ClubLogo team={club} size="lg" />
                      <span>{club}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 pt-0">
          {clubSummary ? (
            <>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Played</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums">{clubSummary.played}</p>
                </div>
                <div className="rounded-lg border border-sky-200 bg-sky-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-sky-700">Points</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums text-sky-700">{clubSummary.actualPoints}</p>
                </div>
                <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-emerald-700">Wins</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums text-emerald-700">{clubSummary.wins}</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Draws</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums">{clubSummary.draws}</p>
                </div>
                <div className="rounded-lg border border-rose-200 bg-rose-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-rose-700">Losses</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums text-rose-700">{clubSummary.losses}</p>
                </div>
                <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-amber-700">Model Accuracy / Confidence</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums text-amber-700">
                    {formatPercent(clubSummary.modelAccuracy)} / {formatPercent(clubSummary.averageConfidence)}
                  </p>
                </div>
              </div>

              <div className="rounded-xl border border-indigo-200 bg-indigo-50/40 p-4">
                <div className="space-y-1">
                  <p className="text-xs uppercase tracking-[0.16em] text-indigo-700">Actual vs Model Expectation</p>
                  <p className="text-sm text-indigo-900/90">Delta is actual minus expected from model probabilities.</p>
                </div>
                <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  <div className="rounded-lg border border-indigo-200 bg-white p-3">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Wins</p>
                    <p className="mt-1 text-sm tabular-nums">
                      {clubSummary.wins} actual vs {clubSummary.expectedWins.toFixed(1)} expected
                    </p>
                    <p className={cn('mt-1 text-sm font-semibold tabular-nums', comparisonDeltaClass(clubSummary.winDelta))}>
                      {formatSigned(clubSummary.winDelta)}
                    </p>
                  </div>
                  <div className="rounded-lg border border-indigo-200 bg-white p-3">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Draws</p>
                    <p className="mt-1 text-sm tabular-nums">
                      {clubSummary.draws} actual vs {clubSummary.expectedDraws.toFixed(1)} expected
                    </p>
                    <p className={cn('mt-1 text-sm font-semibold tabular-nums', comparisonDeltaClass(clubSummary.drawDelta))}>
                      {formatSigned(clubSummary.drawDelta)}
                    </p>
                  </div>
                  <div className="rounded-lg border border-indigo-200 bg-white p-3">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Losses</p>
                    <p className="mt-1 text-sm tabular-nums">
                      {clubSummary.losses} actual vs {clubSummary.expectedLosses.toFixed(1)} expected
                    </p>
                    <p className={cn('mt-1 text-sm font-semibold tabular-nums', comparisonDeltaClass(clubSummary.lossDelta, true))}>
                      {formatSigned(clubSummary.lossDelta)}
                    </p>
                  </div>
                  <div className="rounded-lg border border-indigo-200 bg-white p-3">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Points</p>
                    <p className="mt-1 text-sm tabular-nums">
                      {clubSummary.actualPoints} actual vs {clubSummary.expectedPoints.toFixed(1)} expected
                    </p>
                    <p className={cn('mt-1 text-sm font-semibold tabular-nums', comparisonDeltaClass(clubSummary.pointDelta))}>
                      {formatSigned(clubSummary.pointDelta)}
                    </p>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">No fixtures available for this club in the current dataset.</p>
          )}

          <div className="overflow-x-auto overflow-y-hidden rounded-lg border border-slate-200">
            <Table className="min-w-[960px]">
              <TableHeader className="bg-slate-50">
                <TableRow className="border-b-0 hover:bg-transparent">
                  <TableHead>Date</TableHead>
                  <TableHead>Opponent</TableHead>
                  <TableHead>Venue</TableHead>
                  <TableHead>Result</TableHead>
                  <TableHead>Model Pick</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead className="text-right">Win %</TableHead>
                  <TableHead className="text-right">Draw %</TableHead>
                  <TableHead className="text-right">Loss %</TableHead>
                  <TableHead>Prediction</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {clubFixtures.map((fixture) => (
                  <TableRow key={fixture.id} className="h-14">
                    <TableCell className="font-medium tabular-nums">{fixture.matchDate}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <ClubLogo team={fixture.opponent} />
                        <span>{fixture.opponent}</span>
                      </div>
                    </TableCell>
                    <TableCell>{fixture.venue}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className={cn('uppercase tracking-[0.12em]', outcomeBadgeClass(fixture.actualOutcome))}>
                        {outcomeLabelMap[fixture.actualOutcome]}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className={cn('uppercase tracking-[0.12em]', outcomeBadgeClass(fixture.modelOutcome))}>
                        {outcomeLabelMap[fixture.modelOutcome]}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <span
                        className={cn(
                          'inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold tabular-nums',
                          confidenceBadgeClass(fixture.modelConfidence)
                        )}
                      >
                        {formatPercent(fixture.modelConfidence)}
                      </span>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{formatPercent(fixture.winProbability)}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatPercent(fixture.drawProbability)}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatPercent(fixture.lossProbability)}</TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={cn(
                          'uppercase tracking-[0.12em]',
                          fixture.predictionCorrect
                            ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                            : 'border-slate-200 bg-slate-100 text-slate-700'
                        )}
                      >
                        {fixture.predictionCorrect ? 'Correct' : 'Miss'}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </section>
  )
}

function AppNavigation() {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/90 backdrop-blur-sm">
      <div className="mx-auto w-full max-w-[1320px] px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-3 py-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Premier League Predictor</p>
            <p className="text-lg font-semibold tracking-tight text-slate-900">Data-driven match insights</p>
          </div>
          <nav className="flex gap-2 overflow-x-auto pb-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  cn(
                    'rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.12em] transition-colors',
                    isActive
                      ? 'border-slate-900 bg-slate-900 text-white'
                      : 'border-slate-300 bg-white text-slate-700 hover:border-slate-400 hover:text-slate-900'
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </div>
    </header>
  )
}

export default function App() {
  const [selectedGameweek, setSelectedGameweek] = useState(24)
  const [favoriteTeam, setFavoriteTeam] = useState(teamList[0].team)
  const [selectedClub, setSelectedClub] = useState(normalizeTeamName(teamList[0].team))

  const predictionFixtures = useMemo(() => parsePredictionFixtures(premOddsPredictionsRaw), [])
  const modelOutputTable = useMemo(() => buildModelOutputTable(predictionFixtures), [predictionFixtures])

  const availableClubs = useMemo(() => {
    const clubs = new Set()
    predictionFixtures.forEach((fixture) => {
      clubs.add(fixture.homeTeam)
      clubs.add(fixture.awayTeam)
    })
    return [...clubs].sort((a, b) => a.localeCompare(b))
  }, [predictionFixtures])

  const activeClub = availableClubs.includes(selectedClub) ? selectedClub : (availableClubs[0] ?? '')

  const clubFixtures = useMemo(() => {
    if (!activeClub) return []

    return predictionFixtures
      .filter((match) => match.homeTeam === activeClub || match.awayTeam === activeClub)
      .map((match, index) => {
        const isHome = match.homeTeam === activeClub
        const modelPickCode = deriveModelPickCode(match)
        const modelConfidence =
          modelPickCode === 'H'
            ? match.modelHomeProb
            : modelPickCode === 'D'
              ? match.modelDrawProb
              : match.modelAwayProb
        const actualOutcome = outcomeForClub(match.fullTimeResult, isHome)
        const modelOutcome = outcomeForClub(modelPickCode, isHome)

        return {
          id: `${match.id}-${index}`,
          matchDate: match.matchDate,
          opponent: isHome ? match.awayTeam : match.homeTeam,
          venue: isHome ? 'Home' : 'Away',
          actualOutcome,
          modelOutcome,
          modelConfidence,
          winProbability: isHome ? match.modelHomeProb : match.modelAwayProb,
          drawProbability: match.modelDrawProb,
          lossProbability: isHome ? match.modelAwayProb : match.modelHomeProb,
          predictionCorrect: actualOutcome === modelOutcome,
        }
      })
      .sort((a, b) => a.matchDate.localeCompare(b.matchDate))
  }, [activeClub, predictionFixtures])

  const clubSummary = useMemo(() => {
    if (!clubFixtures.length) return null

    const wins = clubFixtures.filter((fixture) => fixture.actualOutcome === 'W').length
    const draws = clubFixtures.filter((fixture) => fixture.actualOutcome === 'D').length
    const losses = clubFixtures.filter((fixture) => fixture.actualOutcome === 'L').length
    const modelHits = clubFixtures.filter((fixture) => fixture.predictionCorrect).length
    const confidenceTotal = clubFixtures.reduce((sum, fixture) => sum + fixture.modelConfidence, 0)
    const expectedWins = clubFixtures.reduce((sum, fixture) => sum + fixture.winProbability, 0)
    const expectedDraws = clubFixtures.reduce((sum, fixture) => sum + fixture.drawProbability, 0)
    const expectedLosses = clubFixtures.reduce((sum, fixture) => sum + fixture.lossProbability, 0)
    const actualPoints = wins * 3 + draws
    const expectedPoints = expectedWins * 3 + expectedDraws

    return {
      played: clubFixtures.length,
      wins,
      draws,
      losses,
      actualPoints,
      expectedWins,
      expectedDraws,
      expectedLosses,
      expectedPoints,
      winDelta: wins - expectedWins,
      drawDelta: draws - expectedDraws,
      lossDelta: losses - expectedLosses,
      pointDelta: actualPoints - expectedPoints,
      modelAccuracy: modelHits / clubFixtures.length,
      averageConfidence: confidenceTotal / clubFixtures.length,
    }
  }, [clubFixtures])

  const { currentTable, predictedTable, topOver, topUnder, favoriteSnapshot } = useMemo(() => {
    const currentRows = buildStandings(selectedGameweek)
    const predictedRows = buildStandings(selectedGameweek)

    const currentTable = [...currentRows]
      .sort(sortByNumber('points'))
      .map((row, index) => ({ ...row, position: index + 1 }))

    const predictedTable = [...predictedRows]
      .sort(sortByNumber('predictedPoints'))
      .map((row, index) => ({
        ...row,
        position: index + 1,
        delta: row.predictedPoints - row.points,
      }))

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
  }, [favoriteTeam, selectedGameweek])

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <AppNavigation />
      <main className="relative z-10 mx-auto w-full max-w-[1440px] px-3 py-8 sm:px-4 md:py-10 lg:px-6 lg:py-12">
        <Routes>
          <Route
            path="/"
            element={
              <OverviewPage
                topOver={topOver}
                topUnder={topUnder}
                favoriteTeam={favoriteTeam}
                favoriteSnapshot={favoriteSnapshot}
                onFavoriteTeamChange={setFavoriteTeam}
              />
            }
          />
          <Route
            path="/tables"
            element={
              <TablesPage
                currentTable={currentTable}
                predictedTable={predictedTable}
                favoriteTeam={favoriteTeam}
                gameweek={selectedGameweek}
                onGameweekChange={setSelectedGameweek}
              />
            }
          />
          <Route
            path="/club"
            element={
              <ClubPage
                clubs={availableClubs}
                selectedClub={activeClub}
                onSelectedClubChange={setSelectedClub}
                clubFixtures={clubFixtures}
                clubSummary={clubSummary}
              />
            }
          />
          <Route path="/methodology" element={<MethodologyPage />} />
          <Route
            path="/model-output"
            element={<ModelOutputPage favoriteTeam={favoriteTeam} modelOutputTable={modelOutputTable} />}
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
