import { useMemo } from 'react'
import { NavLink } from 'react-router-dom'
import { Activity, BrainCircuit, Calendar, Home, MessageSquareQuote } from 'lucide-react'

import { Badge } from '../components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select'
import { ClubLogo } from '../components/shared/ClubLogo'
import { SeasonSelector } from '../components/shared/SeasonSelector'
import { PerformerList } from '../components/standings/PerformerList'
import { deriveMarketPickCode } from '../lib/standings'
import { getDisplayTeamName, normalizeTeamName } from '../lib/teamUtils'
import { formatPercent, formatSigned } from '../lib/formatters'
import { average, buildMatchPageInsightData, probabilityBarHeight } from '../lib/matchInsights'

function OverviewControlPanel({
  season,
  seasonOptions,
  onSeasonChange,
  favoriteTeam,
  onFavoriteTeamChange,
  teamOptions,
}) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <SeasonSelector
        season={season}
        seasonOptions={seasonOptions}
        onSeasonChange={onSeasonChange}
        className="[&_[role=combobox]]:w-full"
      />
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Club</p>
        <Select value={favoriteTeam} onValueChange={onFavoriteTeamChange}>
          <SelectTrigger className="w-full border-slate-300 bg-white text-slate-900">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {teamOptions.map((team) => (
              <SelectItem key={team} value={team}>
                <div className="flex items-center gap-2">
                  <ClubLogo team={team} />
                  <span>{team}</span>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}

function buildOverviewPageData(matches, currentTable, predictedTable) {
  const insightData = buildMatchPageInsightData(matches)
  const clubs = new Set()
  matches.forEach((match) => {
    clubs.add(match.homeTeam)
    clubs.add(match.awayTeam)
  })

  const modelHits = matches.filter((match) => match.predictionCorrect).length
  const averageConfidence = average(matches.map((match) => match.modelConfidence))
  const modelMarketAgreement = matches.length
    ? matches.filter((match) => match.modelPickCode === deriveMarketPickCode(match)).length / matches.length
    : null
  const topEdge = insightData.modelEdges[0] ?? null
  const highestConfidence = [...matches].sort((a, b) => b.modelConfidence - a.modelConfidence)[0] ?? null
  const currentLeader = currentTable[0] ?? null
  const projectedLeader = predictedTable[0] ?? null
  const titleShift =
    currentLeader && projectedLeader && normalizeTeamName(currentLeader.team) !== normalizeTeamName(projectedLeader.team)

  return {
    metrics: [
      { label: 'Fixtures Analysed', value: matches.length.toLocaleString(), detail: `${clubs.size} clubs in this season` },
      {
        label: 'Model Hit Rate',
        value: matches.length ? formatPercent(modelHits / matches.length) : '-',
        detail: `${modelHits}/${matches.length} correct result calls`,
      },
      {
        label: 'Average Confidence',
        value: Number.isFinite(averageConfidence) ? formatPercent(averageConfidence) : '-',
        detail: 'Mean top-outcome probability',
      },
      {
        label: 'Model-Market Agreement',
        value: Number.isFinite(modelMarketAgreement) ? formatPercent(modelMarketAgreement) : '-',
        detail: 'Same 1X2 side as market',
      },
    ],
    topEdge,
    highestConfidence,
    currentLeader,
    projectedLeader,
    titleShift,
  }
}

function OverviewMetricStrip({ metrics }) {
  return (
    <section className="hairline-grid md-cols-2 xl-cols-4 grid gap-0 overflow-hidden rounded-lg bg-white/75 md:grid-cols-2 xl:grid-cols-4">
      {metrics.map((metric) => (
        <div key={metric.label} className="p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{metric.label}</p>
          <p className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">{metric.value}</p>
          <p className="mt-1 text-sm text-slate-600">{metric.detail}</p>
        </div>
      ))}
    </section>
  )
}

function overviewProbabilityRows(match) {
  return [
    { code: 'H', shortLabel: 'Home', model: match.modelHomeProb },
    { code: 'D', shortLabel: 'Draw', model: match.modelDrawProb },
    { code: 'A', shortLabel: 'Away', model: match.modelAwayProb },
  ]
}

function OverviewHeroSignal({ data }) {
  const confidenceRows = data.highestConfidence
    ? overviewProbabilityRows(data.highestConfidence)
    : []
  const leaderRows = [
    { label: 'Current', row: data.currentLeader },
    { label: 'Projected', row: data.projectedLeader },
  ].filter((item) => item.row)

  return (
    <Card className="overflow-hidden border-slate-200 bg-white/85 shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Season Intelligence</p>
            <CardTitle className="mt-2 text-xl">Signal Board</CardTitle>
          </div>
          {data.titleShift && (
            <Badge variant="outline" className="border-amber-200 bg-amber-50 text-amber-700">
              Table divergence
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-5 pt-0">
        <div className="hairline-grid grid grid-cols-2 gap-0 rounded-lg bg-slate-50/70">
          {leaderRows.map((item) => (
            <div key={item.label} className="p-3">
              <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">{item.label} Leader</p>
              <div className="mt-2 flex items-center gap-2">
                <ClubLogo team={item.row.team} />
                <p className="text-sm font-semibold text-slate-900">{getDisplayTeamName(item.row.team)}</p>
              </div>
              <p className="mt-1 text-xs tabular-nums text-slate-600">
                {item.label === 'Current' ? `${item.row.points} pts` : `${item.row.predictedPoints} projected pts`}
              </p>
            </div>
          ))}
        </div>

        {data.highestConfidence && (
          <div>
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Strongest Model Read</p>
                <p className="mt-1 text-sm font-semibold text-slate-900">
                  {getDisplayTeamName(data.highestConfidence.homeTeam)} vs {getDisplayTeamName(data.highestConfidence.awayTeam)}
                </p>
              </div>
              <span className="text-lg font-semibold tabular-nums text-slate-950">
                {formatPercent(data.highestConfidence.modelConfidence)}
              </span>
            </div>
            <div className="mt-3 space-y-2">
              {confidenceRows.map((row) => (
                <div key={`overview-confidence-${row.code}`} className="grid grid-cols-[4.5rem_1fr_3.5rem] items-center gap-2 text-xs">
                  <span className="font-medium text-slate-600">{row.shortLabel}</span>
                  <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                    <div className="h-full rounded-full bg-slate-900" style={{ width: probabilityBarHeight(row.model) }} />
                  </div>
                  <span className="text-right tabular-nums text-slate-700">{formatPercent(row.model)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {data.topEdge && (
          <div className="rounded-lg bg-sky-50/70 p-3">
            <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-sky-700">Largest Model Edge</p>
            <p className="mt-1 text-sm font-semibold text-slate-900">
              {getDisplayTeamName(data.topEdge.homeTeam)} vs {getDisplayTeamName(data.topEdge.awayTeam)}
            </p>
            <p className="mt-1 text-xs text-slate-600">
              {data.topEdge.edgeLabel}: {formatSigned(data.topEdge.edgeDelta * 100, 1)} pts versus market
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function OverviewActionGrid() {
  const actions = [
    {
      to: '/matches',
      title: 'Upcoming Matches',
      description: 'Scan the remaining fixture list by gameweek with model probabilities and table context.',
      icon: Calendar,
    },
    {
      to: '/results',
      title: 'Open Results Lab',
      description: 'Drill into completed fixture probabilities, odds movement, shots, cards, and scoring progression.',
      icon: Activity,
    },
    {
      to: '/model-output',
      title: 'Explore Model Output',
      description: 'Compare model picks with market signals and switch into the predicted season table.',
      icon: BrainCircuit,
    },
    {
      to: '/talking-points',
      title: 'Find Talking Points',
      description: 'Turn model edges, market disagreements, table gaps, and result surprises into user-ready angles.',
      icon: MessageSquareQuote,
    },
    {
      to: '/club',
      title: 'Track a Club',
      description: 'Follow one team through outcomes, confidence, expected points, and prediction accuracy.',
      icon: Home,
    },
  ]

  return (
    <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {actions.map((item) => {
        const Icon = item.icon
        return (
          <NavLink
            key={item.to}
            to={item.to}
            className="group rounded-lg bg-white/75 p-5 transition-colors hover:bg-white"
          >
            <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-slate-900 text-white transition-transform group-hover:-translate-y-0.5">
              <Icon className="h-5 w-5" />
            </div>
            <h3 className="text-base font-semibold text-slate-950">{item.title}</h3>
            <p className="mt-2 text-sm leading-6 text-slate-600">{item.description}</p>
            <p className="mt-4 text-xs font-semibold uppercase tracking-[0.14em] text-slate-500 group-hover:text-slate-900">
              View
            </p>
          </NavLink>
        )
      })}
    </section>
  )
}

export function OverviewPage({
  season,
  seasonOptions,
  onSeasonChange,
  matches,
  currentTable,
  predictedTable,
  topOver,
  topUnder,
  favoriteTeam,
  onFavoriteTeamChange,
  teamOptions,
}) {
  const overviewData = useMemo(
    () => buildOverviewPageData(matches, currentTable, predictedTable),
    [matches, currentTable, predictedTable]
  )

  return (
    <div className="space-y-6">
      <section className="grid items-start gap-6 lg:grid-cols-[1.35fr_0.9fr]">
        <div className="space-y-5">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Premier League Predictor</p>
          <h1 className="max-w-4xl text-4xl font-semibold leading-tight tracking-tight text-slate-950 md:text-6xl">
            Every fixture, every market move, every model signal in one sharp view.
          </h1>
          <p className="max-w-3xl text-lg leading-relaxed text-slate-700">
            Turn Premier League results into a live intelligence layer: match probabilities, bookmaker context,
            pressure stats, confidence levels, and projected tables.
          </p>
          <p className="max-w-3xl text-base leading-relaxed text-muted-foreground md:text-lg">
            The app brings together historical fixture data, pre-match odds, softmax probability modelling, and
            post-match analysis so users can move from a headline scoreline to the underlying signal in seconds.
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
          <div className="flex flex-wrap gap-3 pt-2">
            <NavLink
              to="/model-output"
              className="rounded-full bg-slate-900 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-slate-700"
            >
              Explore model insights
            </NavLink>
            <NavLink
              to="/matches"
              className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 transition-colors hover:border-slate-500"
            >
              View upcoming matches
            </NavLink>
            <NavLink
              to="/results"
              className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 transition-colors hover:border-slate-500"
            >
              Open results
            </NavLink>
            <NavLink
              to="/talking-points"
              className="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 transition-colors hover:border-slate-500"
            >
              View talking points
            </NavLink>
          </div>
        </div>
        <div className="space-y-5">
          <OverviewHeroSignal data={overviewData} />
          <OverviewControlPanel
            season={season}
            seasonOptions={seasonOptions}
            onSeasonChange={onSeasonChange}
            favoriteTeam={favoriteTeam}
            onFavoriteTeamChange={onFavoriteTeamChange}
            teamOptions={teamOptions}
          />
        </div>
      </section>

      <OverviewMetricStrip metrics={overviewData.metrics} />
      <OverviewActionGrid />

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
