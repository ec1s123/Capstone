// This code was generated with Codex.
import { createPortal } from 'react-dom'
import { ChevronLeft, ChevronRight, X } from 'lucide-react'

import { Badge } from '../ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { ClubLogo } from '../shared/ClubLogo'
import { cn } from '../../lib/utils'
import { deriveMarketPickCode } from '../../lib/standings'
import { getDisplayTeamName } from '../../lib/teamUtils'
import {
  comparisonDeltaClass,
  formatMatchOutcome,
  formatOdds,
  formatOptionalPercent,
  formatOptionalStat,
  formatPercent,
  formatProbabilityPointGap,
  formatScoreline,
  formatSigned,
  formatStatPair,
  matchOutcomeBadgeClass,
} from '../../lib/formatters'
import {
  barPercent,
  outcomeProbabilityRows,
  probabilityBarHeight,
  ratioOrNull,
  safeChartValue,
} from '../../lib/matchInsights'

export function MatchOverviewPanel({ match }) {
  const marketPickCode = deriveMarketPickCode(match)
  const halfTimeScore = formatScoreline(match.halfTimeHomeGoals, match.halfTimeAwayGoals)
  const predictionStatus = match.predictionCorrect ? 'Correct' : 'Miss'
  const predictionTone = match.predictionCorrect
    ? 'bg-emerald-50 text-emerald-700'
    : 'bg-rose-50 text-rose-700'

  const overviewItems = [
    { label: 'Date', value: match.matchDate, valueClass: 'text-xl tabular-nums' },
    { label: 'HT', value: halfTimeScore, valueClass: 'text-xl tabular-nums' },
    { label: 'FT', value: formatScoreline(match.homeGoals, match.awayGoals), valueClass: 'text-xl tabular-nums' },
    { label: 'Confidence', value: formatPercent(match.modelConfidence), valueClass: 'text-xl tabular-nums' },
    { label: 'Model', value: formatMatchOutcome(match.modelPickCode, match), badgeClass: matchOutcomeBadgeClass(match.modelPickCode) },
    { label: 'Market', value: formatMatchOutcome(marketPickCode, match), badgeClass: matchOutcomeBadgeClass(marketPickCode) },
    { label: 'Result', value: formatMatchOutcome(match.fullTimeResult, match), badgeClass: matchOutcomeBadgeClass(match.fullTimeResult) },
    { label: 'Prediction', value: predictionStatus, badgeClass: predictionTone, uppercase: true },
  ]

  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardContent className="grid gap-2 p-4 sm:grid-cols-2 lg:grid-cols-4 2xl:grid-cols-8">
        {overviewItems.map((item) => (
          <div key={item.label} className="min-w-0 rounded-lg bg-slate-50/70 p-3">
            <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">{item.label}</p>
            {item.badgeClass ? (
              <Badge
                variant="outline"
                className={cn(
                  'mt-2 max-w-full truncate px-2.5 py-1 font-semibold tracking-normal',
                  item.uppercase && 'uppercase tracking-[0.12em]',
                  item.badgeClass
                )}
              >
                {item.value}
              </Badge>
            ) : (
              <p
                className={cn(
                  'mt-1 truncate font-semibold text-slate-900',
                  item.valueClass ?? 'text-sm'
                )}
              >
                {item.value}
              </p>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

export function ProbabilityComparisonChart({ match }) {
  const rows = outcomeProbabilityRows(match)
  const marketPickCode = deriveMarketPickCode(match)
  const largestModelEdge = rows
    .map((row) => ({ ...row, delta: row.model - row.market }))
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))[0]

  return (
    <Card className="h-full border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle className="text-base">Probability Comparison</CardTitle>
            <CardDescription>Model probabilities against the implied Bet365 market view.</CardDescription>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className={cn('font-semibold tracking-normal', matchOutcomeBadgeClass(match.modelPickCode))}>
              Model: {formatMatchOutcome(match.modelPickCode, match)}
            </Badge>
            <Badge variant="outline" className={cn('font-semibold tracking-normal', matchOutcomeBadgeClass(marketPickCode))}>
              Market: {formatMatchOutcome(marketPickCode, match)}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div className="rounded-lg border border-slate-200 bg-slate-50/60 p-4">
          <div className="flex h-56 items-end gap-4 border-b border-slate-200 pb-3">
            {rows.map((row) => (
              <div key={`probability-${row.code}`} className="flex min-w-0 flex-1 flex-col items-center gap-2">
                <div className="flex h-40 w-full items-end justify-center gap-2">
                  <div className="flex h-full w-8 items-end rounded-t bg-slate-200/80">
                    <div
                      className="w-full rounded-t bg-sky-500"
                      style={{ height: probabilityBarHeight(row.model) }}
                      title={`Model ${formatPercent(row.model)}`}
                    />
                  </div>
                  <div className="flex h-full w-8 items-end rounded-t bg-slate-200/80">
                    <div
                      className="w-full rounded-t bg-amber-500"
                      style={{ height: probabilityBarHeight(row.market) }}
                      title={`Market ${formatPercent(row.market)}`}
                    />
                  </div>
                </div>
                <div className="min-w-0 text-center">
                  <p className="truncate text-xs font-semibold text-slate-800">{row.shortLabel}</p>
                  <p className="text-[11px] text-slate-500">{row.code === match.fullTimeResult ? 'Actual' : 'Outcome'}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-600">
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-sky-500" />
              Model
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-amber-500" />
              Market
            </span>
          </div>
        </div>

        <div className="space-y-3">
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
              Largest Model-Market Probability Gap
            </p>
            <p className="mt-1 text-lg font-semibold text-slate-900">{largestModelEdge.label}</p>
            <p className="text-sm text-slate-600">
              Model rates it {formatProbabilityPointGap(largestModelEdge.delta)}.
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Odds Coverage</p>
            <p className="mt-1 text-lg font-semibold text-slate-900">{match.bookmakerOdds?.length ?? 0} books</p>
            <p className="text-sm text-slate-600">Opening and closing prices are listed below.</p>
          </div>
          <div className="space-y-2">
            {rows.map((row) => (
              <div key={`probability-row-${row.code}`} className="grid grid-cols-[4.5rem_1fr_auto] items-center gap-2 text-xs">
                <span className="truncate font-medium text-slate-700">{row.shortLabel}</span>
                <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full rounded-full bg-sky-500" style={{ width: probabilityBarHeight(row.model) }} />
                </div>
                <span className="tabular-nums text-slate-600">{formatPercent(row.model)}</span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function ShotQualityChart({ match }) {
  const teams = [
    {
      team: match.homeTeam,
      displayTeam: getDisplayTeamName(match.homeTeam),
      goals: match.homeGoals,
      shots: match.homeShots,
      shotsOnTarget: match.homeShotsOnTarget,
      corners: match.homeCorners,
      color: 'bg-sky-500',
      light: 'bg-sky-100',
    },
    {
      team: match.awayTeam,
      displayTeam: getDisplayTeamName(match.awayTeam),
      goals: match.awayGoals,
      shots: match.awayShots,
      shotsOnTarget: match.awayShotsOnTarget,
      corners: match.awayCorners,
      color: 'bg-rose-500',
      light: 'bg-rose-100',
    },
  ]
  const maxVolume = Math.max(...teams.flatMap((team) => [safeChartValue(team.shots), safeChartValue(team.shotsOnTarget), safeChartValue(team.goals)]), 1)

  return (
    <Card className="h-full border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Shot Quality Funnel</CardTitle>
        <CardDescription>Volume, accuracy, and conversion from the available post-match stats.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {teams.map((team) => {
          const targetRate = ratioOrNull(team.shotsOnTarget, team.shots)
          const conversionRate = ratioOrNull(team.goals, team.shots)

          return (
            <div key={`shot-quality-${team.team}`} className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <ClubLogo team={team.team} />
                  <p className="truncate text-sm font-semibold text-slate-900">{team.displayTeam}</p>
                </div>
                <p className="text-lg font-semibold tabular-nums text-slate-900">{formatOptionalStat(team.goals)}</p>
              </div>
              <div className="mt-3 space-y-2">
                {[
                  ['Shots', team.shots],
                  ['On Target', team.shotsOnTarget],
                  ['Goals', team.goals],
                ].map(([label, value]) => (
                  <div key={`${team.team}-${label}`} className="grid grid-cols-[5rem_1fr_2.5rem] items-center gap-2 text-xs">
                    <span className="font-medium text-slate-600">{label}</span>
                    <div className={cn('h-2 overflow-hidden rounded-full', team.light)}>
                      <div className={cn('h-full rounded-full', team.color)} style={{ width: barPercent(value, maxVolume) }} />
                    </div>
                    <span className="text-right tabular-nums text-slate-700">{formatOptionalStat(value)}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
                <div className="rounded-md border border-slate-200 bg-white p-2">
                  <p className="text-slate-500">Target Rate</p>
                  <p className="mt-1 font-semibold tabular-nums text-slate-900">{formatOptionalPercent(targetRate)}</p>
                </div>
                <div className="rounded-md border border-slate-200 bg-white p-2">
                  <p className="text-slate-500">Conversion</p>
                  <p className="mt-1 font-semibold tabular-nums text-slate-900">{formatOptionalPercent(conversionRate)}</p>
                </div>
                <div className="rounded-md border border-slate-200 bg-white p-2">
                  <p className="text-slate-500">Corners</p>
                  <p className="mt-1 font-semibold tabular-nums text-slate-900">{formatOptionalStat(team.corners)}</p>
                </div>
              </div>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

export function MatchShareChart({ match }) {
  const metrics = [
    { label: 'Goals', home: match.homeGoals, away: match.awayGoals },
    { label: 'Shots', home: match.homeShots, away: match.awayShots },
    { label: 'Shots On Target', home: match.homeShotsOnTarget, away: match.awayShotsOnTarget },
    { label: 'Corners', home: match.homeCorners, away: match.awayCorners },
    { label: 'Fouls', home: match.homeFouls, away: match.awayFouls },
    { label: 'Yellow Cards', home: match.homeYellowCards, away: match.awayYellowCards },
    { label: 'Red Cards', home: match.homeRedCards, away: match.awayRedCards },
  ].filter((metric) => Number.isFinite(metric.home) && Number.isFinite(metric.away))

  return (
    <Card className="h-full border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Match Share</CardTitle>
        <CardDescription>Side-by-side share of the main event and discipline stats.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
          <span className="truncate text-sky-700">{getDisplayTeamName(match.homeTeam)}</span>
          <span>Metric</span>
          <span className="truncate text-right text-rose-700">{getDisplayTeamName(match.awayTeam)}</span>
        </div>
        {metrics.map((metric) => {
          const total = safeChartValue(metric.home) + safeChartValue(metric.away)
          const homeShare = total > 0 ? (metric.home / total) * 100 : 50
          const awayShare = total > 0 ? 100 - homeShare : 50
          const delta = metric.home - metric.away

          return (
            <div key={`match-share-${metric.label}`} className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
              <div className="mb-2 flex items-center justify-between gap-3 text-xs">
                <span className="font-semibold tabular-nums text-slate-900">{metric.home}</span>
                <span className="text-center font-semibold text-slate-600">{metric.label}</span>
                <span className="font-semibold tabular-nums text-slate-900">{metric.away}</span>
              </div>
              <div className="flex h-3 overflow-hidden rounded-full bg-slate-100">
                <div className="bg-sky-500" style={{ width: `${homeShare}%` }} />
                <div className="bg-rose-500" style={{ width: `${awayShare}%` }} />
              </div>
              <p className="mt-2 text-center text-[11px] text-slate-500">
                Differential: {formatSigned(delta, 0)}
              </p>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

export function MatchRadarChart({ match }) {
  const axes = [
    { label: 'Goals', home: safeChartValue(match.homeGoals), away: safeChartValue(match.awayGoals) },
    { label: 'Shots', home: safeChartValue(match.homeShots), away: safeChartValue(match.awayShots) },
    { label: 'On Target', home: safeChartValue(match.homeShotsOnTarget), away: safeChartValue(match.awayShotsOnTarget) },
    { label: 'Corners', home: safeChartValue(match.homeCorners), away: safeChartValue(match.awayCorners) },
    {
      label: 'Accuracy',
      home: safeChartValue(ratioOrNull(match.homeShotsOnTarget, match.homeShots)),
      away: safeChartValue(ratioOrNull(match.awayShotsOnTarget, match.awayShots)),
      scale: 1,
    },
  ]
  const center = 120
  const radius = 78
  const pointFor = (index, value) => {
    const axis = axes[index]
    const maxValue = axis.scale ?? Math.max(axis.home, axis.away, 1)
    const normalized = maxValue > 0 ? Math.min(value / maxValue, 1) : 0
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / axes.length
    return {
      x: center + Math.cos(angle) * radius * normalized,
      y: center + Math.sin(angle) * radius * normalized,
    }
  }
  const axisPoint = (index, distance = radius) => {
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / axes.length
    return {
      x: center + Math.cos(angle) * distance,
      y: center + Math.sin(angle) * distance,
    }
  }
  const homePoints = axes.map((axis, index) => pointFor(index, axis.home)).map((point) => `${point.x},${point.y}`).join(' ')
  const awayPoints = axes.map((axis, index) => pointFor(index, axis.away)).map((point) => `${point.x},${point.y}`).join(' ')

  return (
    <Card className="h-full border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Pressure Profile</CardTitle>
        <CardDescription>Normalized comparison across scoring, shot volume, accuracy, and set-piece pressure.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 md:grid-cols-[15rem_1fr]">
        <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
          <svg viewBox="0 0 240 240" role="img" aria-label="Pressure profile radar chart" className="h-60 w-full">
            {[0.33, 0.66, 1].map((ring) => (
              <polygon
                key={`radar-ring-${ring}`}
                points={axes.map((_, index) => {
                  const point = axisPoint(index, radius * ring)
                  return `${point.x},${point.y}`
                }).join(' ')}
                fill="none"
                stroke="#cbd5e1"
                strokeWidth="1"
              />
            ))}
            {axes.map((axis, index) => {
              const end = axisPoint(index)
              const labelPoint = axisPoint(index, radius + 21)
              return (
                <g key={`radar-axis-${axis.label}`}>
                  <line x1={center} y1={center} x2={end.x} y2={end.y} stroke="#cbd5e1" strokeWidth="1" />
                  <text
                    x={labelPoint.x}
                    y={labelPoint.y}
                    textAnchor={Math.abs(labelPoint.x - center) < 10 ? 'middle' : labelPoint.x > center ? 'start' : 'end'}
                    dominantBaseline="middle"
                    className="fill-slate-500 text-[10px] font-semibold"
                  >
                    {axis.label}
                  </text>
                </g>
              )
            })}
            <polygon points={homePoints} fill="rgba(14, 165, 233, 0.24)" stroke="#0ea5e9" strokeWidth="2" />
            <polygon points={awayPoints} fill="rgba(244, 63, 94, 0.2)" stroke="#f43f5e" strokeWidth="2" />
          </svg>
          <div className="flex flex-wrap justify-center gap-3 text-xs text-slate-600">
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-sky-500" />
              {getDisplayTeamName(match.homeTeam)}
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-rose-500" />
              {getDisplayTeamName(match.awayTeam)}
            </span>
          </div>
        </div>
        <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-1">
          {axes.map((axis) => (
            <div key={`radar-value-${axis.label}`} className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">{axis.label}</p>
              <div className="mt-2 flex items-center justify-between gap-3 text-sm font-semibold text-slate-900">
                <span className="truncate text-sky-700">
                  {axis.label === 'Accuracy' ? formatOptionalPercent(axis.home) : formatOptionalStat(axis.home)}
                </span>
                <span className="text-slate-400">vs</span>
                <span className="truncate text-right text-rose-700">
                  {axis.label === 'Accuracy' ? formatOptionalPercent(axis.away) : formatOptionalStat(axis.away)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function ScoreProgressionChart({ match }) {
  const hasHalfTime = Number.isFinite(match.halfTimeHomeGoals) && Number.isFinite(match.halfTimeAwayGoals)
  const phases = hasHalfTime
    ? [
        { label: 'Kickoff', home: 0, away: 0 },
        { label: 'Half Time', home: match.halfTimeHomeGoals, away: match.halfTimeAwayGoals },
        { label: 'Full Time', home: match.homeGoals, away: match.awayGoals },
      ]
    : [
        { label: 'Kickoff', home: 0, away: 0 },
        { label: 'Full Time', home: match.homeGoals, away: match.awayGoals },
      ]
  const maxGoals = Math.max(...phases.flatMap((phase) => [safeChartValue(phase.home), safeChartValue(phase.away)]), 1)
  const width = 320
  const height = 150
  const padding = 24
  const chartWidth = width - padding * 2
  const chartHeight = height - padding * 2
  const point = (phase, index) => {
    const x = padding + (index / Math.max(phases.length - 1, 1)) * chartWidth
    const homeY = padding + chartHeight - (safeChartValue(phase.home) / maxGoals) * chartHeight
    const awayY = padding + chartHeight - (safeChartValue(phase.away) / maxGoals) * chartHeight
    return { x, homeY, awayY }
  }
  const plotted = phases.map(point)
  const homeSecondHalf = hasHalfTime && Number.isFinite(match.homeGoals) ? match.homeGoals - match.halfTimeHomeGoals : null
  const awaySecondHalf = hasHalfTime && Number.isFinite(match.awayGoals) ? match.awayGoals - match.halfTimeAwayGoals : null

  return (
    <Card className="h-full border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Scoring Progression</CardTitle>
        <CardDescription>Half-time and full-time score movement where the fixture has interval data.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
          <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Score progression chart" className="h-44 w-full">
            <line x1={padding} y1={padding + chartHeight} x2={padding + chartWidth} y2={padding + chartHeight} stroke="#cbd5e1" />
            {[...Array(maxGoals + 1)].map((_, value) => {
              const y = padding + chartHeight - (value / maxGoals) * chartHeight
              return (
                <g key={`score-grid-${value}`}>
                  <line x1={padding} y1={y} x2={padding + chartWidth} y2={y} stroke="#e2e8f0" strokeWidth="1" />
                  <text x={8} y={y + 3} className="fill-slate-400 text-[10px]">
                    {value}
                  </text>
                </g>
              )
            })}
            <polyline
              points={plotted.map((plot) => `${plot.x},${plot.homeY}`).join(' ')}
              fill="none"
              stroke="#0ea5e9"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <polyline
              points={plotted.map((plot) => `${plot.x},${plot.awayY}`).join(' ')}
              fill="none"
              stroke="#f43f5e"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            {plotted.map((plot, index) => (
              <g key={`score-point-${phases[index].label}`}>
                <circle cx={plot.x} cy={plot.homeY} r="4" fill="#0ea5e9" />
                <circle cx={plot.x} cy={plot.awayY} r="4" fill="#f43f5e" />
                <text x={plot.x} y={height - 5} textAnchor="middle" className="fill-slate-500 text-[10px] font-semibold">
                  {phases[index].label}
                </text>
              </g>
            ))}
          </svg>
          <div className="mt-2 flex flex-wrap justify-center gap-3 text-xs text-slate-600">
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-sky-500" />
              {getDisplayTeamName(match.homeTeam)}
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-rose-500" />
              {getDisplayTeamName(match.awayTeam)}
            </span>
          </div>
        </div>
        <div className="grid gap-2 sm:grid-cols-3">
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Half Time</p>
            <p className="mt-1 text-lg font-semibold tabular-nums text-slate-900">
              {formatScoreline(match.halfTimeHomeGoals, match.halfTimeAwayGoals)}
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Second Half</p>
            <p className="mt-1 text-lg font-semibold tabular-nums text-slate-900">
              {formatScoreline(homeSecondHalf, awaySecondHalf)}
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Full Time</p>
            <p className="mt-1 text-lg font-semibold tabular-nums text-slate-900">
              {formatScoreline(match.homeGoals, match.awayGoals)}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function BookmakerOddsPanel({ match }) {
  const bookmakerOdds = Array.isArray(match.bookmakerOdds)
    ? match.bookmakerOdds.filter((bookmaker) =>
        [
          bookmaker.home,
          bookmaker.draw,
          bookmaker.away,
          bookmaker.closingHome,
          bookmaker.closingDraw,
          bookmaker.closingAway,
        ].some((value) => Number.isFinite(value) && value > 0)
      )
    : []
  const oddsGridStyle = {
    gridTemplateColumns: 'minmax(14rem, 1.55fr) repeat(6, minmax(7.5rem, 1fr))',
  }

  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <CardTitle className="text-base">Bookmaker Odds</CardTitle>
            <CardDescription>
              Opening and closing 1X2 odds from every bookmaker column available for this fixture.
            </CardDescription>
          </div>
          <Badge variant="outline" className="w-fit border-slate-200 bg-slate-50 text-slate-700">
            {bookmakerOdds.length} sources
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        {bookmakerOdds.length ? (
          <div className="matches-scroll-container overflow-x-auto rounded-lg bg-white">
            <div className="min-w-[1020px] text-sm" role="table" aria-label="Bookmaker odds">
              <div
                className="grid items-center bg-slate-50 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500"
                style={oddsGridStyle}
                role="row"
              >
                <div className="px-4 py-3 text-left" role="columnheader">Bookmaker</div>
                <div className="px-4 py-3 text-right" role="columnheader">Open Home</div>
                <div className="px-4 py-3 text-right" role="columnheader">Open Draw</div>
                <div className="px-4 py-3 text-right" role="columnheader">Open Away</div>
                <div className="px-4 py-3 text-right" role="columnheader">Close Home</div>
                <div className="px-4 py-3 text-right" role="columnheader">Close Draw</div>
                <div className="px-4 py-3 text-right" role="columnheader">Close Away</div>
              </div>
              <div className="divide-y divide-slate-200 bg-white" role="rowgroup">
                {bookmakerOdds.map((bookmaker) => (
                  <div
                    key={`bookmaker-odds-${bookmaker.code}`}
                    className="grid items-center"
                    style={oddsGridStyle}
                    role="row"
                  >
                    <div className="min-w-0 px-4 py-3 text-left font-semibold text-slate-900" role="cell">
                      <div className="flex min-w-0 items-baseline gap-2">
                        <span className="truncate">{bookmaker.label}</span>
                        <span className="shrink-0 text-xs font-medium text-slate-400">{bookmaker.code}</span>
                      </div>
                    </div>
                    <div className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-700" role="cell">
                      {formatOdds(bookmaker.home)}
                    </div>
                    <div className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-700" role="cell">
                      {formatOdds(bookmaker.draw)}
                    </div>
                    <div className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-700" role="cell">
                      {formatOdds(bookmaker.away)}
                    </div>
                    <div className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-900" role="cell">
                      {formatOdds(bookmaker.closingHome)}
                    </div>
                    <div className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-900" role="cell">
                      {formatOdds(bookmaker.closingDraw)}
                    </div>
                    <div className="whitespace-nowrap px-4 py-3 text-right tabular-nums text-slate-900" role="cell">
                      {formatOdds(bookmaker.closingAway)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-4 text-sm text-slate-600">
            No bookmaker odds are available for this fixture.
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export function MatchDetailsDrawer({ matches, activeIndex, onClose, onSelectIndex }) {
  if (activeIndex < 0 || activeIndex >= matches.length) return null

  const match = matches[activeIndex]
  const hasPrevious = activeIndex > 0
  const hasNext = activeIndex < matches.length - 1

  const drawer = (
    <div className="fixed inset-0 z-[100]">
      <button
        type="button"
        className="absolute inset-0 bg-slate-900/55"
        aria-label="Close details"
        onClick={onClose}
      />
      <aside className="absolute inset-2 overflow-hidden rounded-xl border border-slate-200 bg-slate-50 shadow-2xl sm:inset-4 lg:inset-6">
        <div className="flex h-full flex-col">
          <div className="flex flex-col gap-4 border-b border-slate-200 bg-white px-4 py-4 sm:px-5 lg:flex-row lg:items-center lg:justify-between lg:px-6">
            <div className="min-w-0">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Match Analysis</p>
              <div className="mt-2 flex min-w-0 flex-wrap items-center gap-3">
                <ClubLogo team={match.homeTeam} size="lg" />
                <div className="min-w-0">
                  <p className="truncate text-xl font-semibold tracking-tight text-slate-900 sm:text-2xl">
                    {getDisplayTeamName(match.homeTeam)} vs {getDisplayTeamName(match.awayTeam)}
                  </p>
                  <p className="mt-1 text-sm text-slate-600">
                    {match.matchDate} · {formatScoreline(match.homeGoals, match.awayGoals)}
                  </p>
                </div>
                <ClubLogo team={match.awayTeam} size="lg" />
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-300 bg-white text-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
                onClick={() => hasPrevious && onSelectIndex(activeIndex - 1)}
                disabled={!hasPrevious}
                aria-label="Previous match"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-300 bg-white text-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
                onClick={() => hasNext && onSelectIndex(activeIndex + 1)}
                disabled={!hasNext}
                aria-label="Next match"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-300 bg-white text-slate-700"
                onClick={onClose}
                aria-label="Close drawer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div className="drawer-scroll flex-1 overflow-y-auto p-4 sm:p-5 lg:p-6">
            <div className="mx-auto max-w-[1500px] space-y-4">
              <MatchOverviewPanel match={match} />
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(360px,0.75fr)]">
                <ProbabilityComparisonChart match={match} />
                <ScoreProgressionChart match={match} />
              </div>
              <BookmakerOddsPanel match={match} />
              <div className="grid gap-4 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
                <ShotQualityChart match={match} />
                <MatchRadarChart match={match} />
              </div>
              <MatchShareChart match={match} />
            </div>
          </div>
        </div>
      </aside>
    </div>
  )

  if (typeof document === 'undefined') return drawer
  return createPortal(drawer, document.body)
}
