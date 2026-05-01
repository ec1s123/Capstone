import { useEffect, useMemo, useState } from 'react'
import { Activity, Calendar, Gauge, Scale, TrendingUp } from 'lucide-react'

import { Badge } from '../components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table'
import { ClubLogo } from '../components/shared/ClubLogo'
import { FormChips } from '../components/shared/FormChips'
import { SeasonSelector } from '../components/shared/SeasonSelector'
import { cn } from '../lib/utils'
import {
  confidenceBadgeClass,
  formatMatchOutcome,
  formatPercent,
  formatSigned,
  matchOutcomeBadgeClass,
} from '../lib/formatters'

const dateFormatter = new Intl.DateTimeFormat('en-GB', {
  weekday: 'short',
  day: 'numeric',
  month: 'short',
  year: 'numeric',
  timeZone: 'UTC',
})

function formatFixtureDate(dateValue) {
  const parsed = new Date(`${dateValue}T00:00:00Z`)
  if (Number.isNaN(parsed.getTime())) return dateValue
  return dateFormatter.format(parsed)
}

function formatKickoff(match) {
  return `${match.kickoffTime ?? 'TBC'} ${match.timezone ?? ''}`.trim()
}

function probabilityRows(match) {
  return [
    { code: 'H', label: match.homeTeam, shortLabel: 'Home', value: match.modelHomeProb },
    { code: 'D', label: 'Draw', shortLabel: 'Draw', value: match.modelDrawProb },
    { code: 'A', label: match.awayTeam, shortLabel: 'Away', value: match.modelAwayProb },
  ]
}

function groupMatches(matches) {
  const roundMap = new Map()

  matches.forEach((match) => {
    const roundKey = match.roundLabel
    if (!roundMap.has(roundKey)) {
      roundMap.set(roundKey, {
        key: roundKey,
        label: roundKey,
        gameweek: match.gameweek,
        matches: [],
        dateGroups: new Map(),
      })
    }

    const roundGroup = roundMap.get(roundKey)
    roundGroup.matches.push(match)
    if (!roundGroup.dateGroups.has(match.matchDate)) {
      roundGroup.dateGroups.set(match.matchDate, [])
    }
    roundGroup.dateGroups.get(match.matchDate).push(match)
  })

  return [...roundMap.values()]
    .map((roundGroup) => ({
      ...roundGroup,
      firstMatch: roundGroup.matches[0],
      dateGroups: [...roundGroup.dateGroups.entries()].map(([date, dateMatches]) => ({
        date,
        matches: dateMatches,
      })),
    }))
    .sort((a, b) => a.firstMatch.matchDate.localeCompare(b.firstMatch.matchDate))
}

function buildRoundOptions(matches) {
  return groupMatches(matches).map((group) => ({
    key: group.key,
    label: group.label,
    detail: formatFixtureDate(group.firstMatch.matchDate),
  }))
}

function UpcomingInsightCard({ title, description, match, icon: Icon, metric }) {
  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="text-base text-slate-900">{title}</CardTitle>
            <CardDescription>{description}</CardDescription>
          </div>
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-slate-200 bg-slate-100 text-slate-600">
            <Icon className="h-4 w-4" />
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-1 pt-0">
        {match ? (
          <>
            <p className="text-sm font-semibold text-slate-900">
              {match.homeTeam} vs {match.awayTeam}
            </p>
            <p className="text-xs text-muted-foreground">
              {formatFixtureDate(match.matchDate)} · {formatKickoff(match)}
            </p>
            <p className="text-xs text-slate-700">{metric(match)}</p>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">No upcoming fixtures available.</p>
        )}
      </CardContent>
    </Card>
  )
}

function TeamFixtureCell({ team, align = 'left' }) {
  return (
    <div className={cn('flex min-w-0 items-center gap-2', align === 'right' && 'justify-end')}>
      {align === 'right' && <span className="min-w-0 truncate text-right font-semibold text-slate-900">{team}</span>}
      <ClubLogo team={team} />
      {align !== 'right' && <span className="min-w-0 truncate font-semibold text-slate-900">{team}</span>}
    </div>
  )
}

function ProbabilityBars({ match }) {
  return (
    <div className="space-y-2">
      {probabilityRows(match).map((row) => (
        <div key={`${match.id}-${row.code}`} className="grid grid-cols-[3.75rem_1fr_3.25rem] items-center gap-2 text-xs">
          <span className="truncate font-medium text-slate-600">{row.shortLabel}</span>
          <div className="h-2 overflow-hidden rounded-full bg-slate-100">
            <div
              className={cn(
                'h-full rounded-full',
                row.code === 'H' ? 'bg-emerald-500' : row.code === 'A' ? 'bg-rose-500' : 'bg-slate-500'
              )}
              style={{ width: formatPercent(row.value) }}
            />
          </div>
          <span className="text-right tabular-nums text-slate-700">{formatPercent(row.value)}</span>
        </div>
      ))}
    </div>
  )
}

function UpcomingMatchDetails({ match }) {
  if (!match) {
    return (
      <Card className="border-slate-200 bg-white shadow-sm">
        <CardContent className="p-5 text-sm text-slate-600">No upcoming match selected.</CardContent>
      </Card>
    )
  }

  return (
    <Card className="border-slate-200 bg-white shadow-sm min-[1400px]:sticky min-[1400px]:top-28">
      <CardHeader className="p-4 pb-2">
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="text-lg text-slate-900">Match Outlook</CardTitle>
            <CardDescription>
              {match.roundLabel} · {formatFixtureDate(match.matchDate)} · {formatKickoff(match)}
            </CardDescription>
          </div>
          <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-700">
            {match.signalLabel}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 p-4 pt-0">
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
          <TeamFixtureCell team={match.homeTeam} align="right" />
          <span className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">vs</span>
          <TeamFixtureCell team={match.awayTeam} />
        </div>

        <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Model Pick</p>
            <Badge variant="outline" className={cn('font-semibold tracking-normal', matchOutcomeBadgeClass(match.modelPickCode))}>
              {formatMatchOutcome(match.modelPickCode, match)}
            </Badge>
          </div>
          <p className="mt-2 text-2xl font-semibold tabular-nums text-slate-900">
            {formatPercent(match.modelConfidence)}
          </p>
          <p className="text-xs text-slate-600">Top-outcome confidence</p>
        </div>

        <ProbabilityBars match={match} />

        <div className="grid gap-2 sm:grid-cols-2 min-[1400px]:grid-cols-1 2xl:grid-cols-2">
          {[match.homeProfile, match.awayProfile].map((profile) => (
            <div key={`profile-${profile.team}`} className="rounded-lg border border-slate-200 bg-slate-50/70 p-2.5">
              <div className="flex items-center gap-2">
                <ClubLogo team={profile.team} />
                <p className="truncate text-sm font-semibold text-slate-900">{profile.team}</p>
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                <div>
                  <p className="text-slate-500">Table</p>
                  <p className="font-semibold text-slate-900">{profile.actualPosition ?? '-'} · {profile.actualPoints} pts</p>
                </div>
                <div>
                  <p className="text-slate-500">xPts/G</p>
                  <p className="font-semibold tabular-nums text-slate-900">{profile.expectedPointsPerMatch.toFixed(2)}</p>
                </div>
              </div>
              <div className="mt-2">
                <FormChips results={profile.form} />
              </div>
            </div>
          ))}
        </div>

        <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-2.5">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Expected Points Split</p>
          <div className="mt-2 grid grid-cols-[1fr_auto_1fr] items-center gap-3 text-sm">
            <span className="truncate font-semibold text-emerald-700">{match.homeTeam}</span>
            <span className="text-xs text-slate-400">vs</span>
            <span className="truncate text-right font-semibold text-rose-700">{match.awayTeam}</span>
            <span className="tabular-nums text-slate-900">{match.homeExpectedPoints.toFixed(2)}</span>
            <span />
            <span className="text-right tabular-nums text-slate-900">{match.awayExpectedPoints.toFixed(2)}</span>
          </div>
        </div>

        <div className="space-y-2">
          {match.notes.map((note) => (
            <div key={note} className="rounded-lg border border-slate-200 bg-white p-2.5 text-sm leading-5 text-slate-700">
              {note}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function MatchesPage({
  season,
  seasonOptions,
  onSeasonChange,
  matches,
  insights,
}) {
  const [activeMatchId, setActiveMatchId] = useState(matches[0]?.id ?? '')
  const [selectedRound, setSelectedRound] = useState('all')

  const roundOptions = useMemo(() => buildRoundOptions(matches), [matches])
  const visibleMatches = useMemo(
    () => (selectedRound === 'all' ? matches : matches.filter((match) => match.roundLabel === selectedRound)),
    [matches, selectedRound]
  )
  const groupedMatches = useMemo(() => groupMatches(visibleMatches), [visibleMatches])
  const activeMatch = visibleMatches.find((match) => match.id === activeMatchId) ?? visibleMatches[0] ?? matches[0] ?? null

  useEffect(() => {
    if (!matches.length) {
      setActiveMatchId('')
      return
    }
    if (!matches.some((match) => match.id === activeMatchId)) {
      setActiveMatchId(matches[0].id)
    }
  }, [activeMatchId, matches])

  useEffect(() => {
    if (selectedRound === 'all') return
    if (!roundOptions.some((option) => option.key === selectedRound)) {
      setSelectedRound('all')
    }
  }, [roundOptions, selectedRound])

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="space-y-1">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Matches</p>
          <h2 className="text-2xl font-semibold tracking-tight text-slate-900">Upcoming Fixtures and Model Outlook</h2>
          <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
            Remaining Premier League fixtures grouped by gameweek and date, with model probabilities and matchup context.
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div className="w-48 space-y-1">
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Gameweek</p>
            <Select value={selectedRound} onValueChange={setSelectedRound}>
              <SelectTrigger>
                <SelectValue placeholder="Gameweek" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All remaining</SelectItem>
                {roundOptions.map((option) => (
                  <SelectItem key={option.key} value={option.key}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <SeasonSelector season={season} seasonOptions={seasonOptions} onSeasonChange={onSeasonChange} />
        </div>
      </div>

      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <UpcomingInsightCard
          title="Next Match"
          description="Earliest kickoff"
          match={insights.nextMatch}
          icon={Calendar}
          metric={(match) => `Model: ${formatMatchOutcome(match.modelPickCode, match)} (${formatPercent(match.modelConfidence)})`}
        />
        <UpcomingInsightCard
          title="Strongest Lean"
          description="Highest top probability"
          match={insights.strongestLean}
          icon={TrendingUp}
          metric={(match) => `${formatMatchOutcome(match.modelPickCode, match)} at ${formatPercent(match.modelConfidence)}`}
        />
        <UpcomingInsightCard
          title="Closest Call"
          description="Lowest confidence"
          match={insights.closestCall}
          icon={Scale}
          metric={(match) => `${formatSigned(match.expectedPointsGap, 2)} home xPts edge`}
        />
        <UpcomingInsightCard
          title="Biggest Table Gap"
          description="Current-table separation"
          match={insights.biggestTableGap}
          icon={Gauge}
          metric={(match) => `${match.homePositionText} vs ${match.awayPositionText}`}
        />
      </section>

      <div className="grid gap-3 min-[1400px]:grid-cols-[minmax(0,1fr)_22rem] 2xl:grid-cols-[minmax(0,1fr)_23rem]">
        <div className="space-y-3">
          {groupedMatches.map((roundGroup) => (
            <Card key={roundGroup.key} className="border-slate-200 bg-white shadow-sm">
              <CardHeader className="p-4 pb-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <CardTitle className="text-lg text-slate-900">{roundGroup.label}</CardTitle>
                    <CardDescription>{roundGroup.matches.length} fixtures</CardDescription>
                  </div>
                  <Badge variant="outline" className="border-slate-200 bg-slate-50 text-slate-700">
                    GW {roundGroup.gameweek}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-3 p-4 pt-0">
                {roundGroup.dateGroups.map((dateGroup) => (
                  <div key={`${roundGroup.key}-${dateGroup.date}`} className="space-y-1.5">
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-700">
                      <Activity className="h-4 w-4 text-slate-400" />
                      {formatFixtureDate(dateGroup.date)}
                    </div>
                    <div className="matches-scroll-container overflow-hidden rounded-lg border border-slate-200">
                      <Table className="w-full table-fixed text-[13px]">
                        <colgroup>
                          <col className="w-[9%]" />
                          <col className="w-[19%]" />
                          <col className="w-[16%]" />
                          <col className="w-[17%]" />
                          <col className="w-[10%]" />
                          <col className="w-[6.5%]" />
                          <col className="w-[6.5%]" />
                          <col className="w-[6.5%]" />
                          <col className="w-[9.5%]" />
                        </colgroup>
                        <TableHeader className="bg-slate-50">
                          <TableRow className="border-b-0 hover:bg-transparent">
                            <TableHead className="h-9 px-2 first:pl-3">Time</TableHead>
                            <TableHead className="h-9 px-2">Home</TableHead>
                            <TableHead className="h-9 px-2">Away</TableHead>
                            <TableHead className="h-9 px-2">Model Pick</TableHead>
                            <TableHead className="h-9 px-2">Confidence</TableHead>
                            <TableHead className="h-9 px-2 text-right">Home</TableHead>
                            <TableHead className="h-9 px-2 text-right">Draw</TableHead>
                            <TableHead className="h-9 px-2 text-right">Away</TableHead>
                            <TableHead className="h-9 px-2 last:pr-3">Signal</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {dateGroup.matches.map((match) => {
                            const pickLabel = formatMatchOutcome(match.modelPickCode, match)
                            return (
                              <TableRow
                                key={match.id}
                                className={cn(
                                  'cursor-pointer hover:bg-slate-50',
                                  activeMatch?.id === match.id && 'bg-sky-50/70 hover:bg-sky-50'
                                )}
                                onClick={() => setActiveMatchId(match.id)}
                              >
                                <TableCell className="whitespace-nowrap px-2 py-2.5 pl-3 font-medium tabular-nums text-slate-700">
                                  {formatKickoff(match)}
                                </TableCell>
                                <TableCell className="min-w-0 px-2 py-2.5">
                                  <TeamFixtureCell team={match.homeTeam} />
                                </TableCell>
                                <TableCell className="min-w-0 px-2 py-2.5">
                                  <TeamFixtureCell team={match.awayTeam} />
                                </TableCell>
                                <TableCell className="px-2 py-2.5">
                                  <Badge
                                    variant="outline"
                                    title={pickLabel}
                                    className={cn(
                                      'max-w-full truncate whitespace-nowrap px-2 font-semibold tracking-normal',
                                      matchOutcomeBadgeClass(match.modelPickCode)
                                    )}
                                  >
                                    {pickLabel}
                                  </Badge>
                                </TableCell>
                                <TableCell className="px-2 py-2.5">
                                  <span
                                    className={cn(
                                      'inline-flex max-w-full rounded-full border px-2 py-0.5 text-xs font-semibold tabular-nums',
                                      confidenceBadgeClass(match.modelConfidence)
                                    )}
                                  >
                                    {formatPercent(match.modelConfidence)}
                                  </span>
                                </TableCell>
                                <TableCell className="whitespace-nowrap px-2 py-2.5 text-right tabular-nums text-slate-700">
                                  {formatPercent(match.modelHomeProb)}
                                </TableCell>
                                <TableCell className="whitespace-nowrap px-2 py-2.5 text-right tabular-nums text-slate-700">
                                  {formatPercent(match.modelDrawProb)}
                                </TableCell>
                                <TableCell className="whitespace-nowrap px-2 py-2.5 text-right tabular-nums text-slate-700">
                                  {formatPercent(match.modelAwayProb)}
                                </TableCell>
                                <TableCell className="px-2 py-2.5 pr-3">
                                  <Badge
                                    variant="outline"
                                    title={match.signalLabel}
                                    className="max-w-full truncate whitespace-nowrap border-slate-200 bg-white px-2 text-slate-700"
                                  >
                                    {match.signalLabel}
                                  </Badge>
                                </TableCell>
                              </TableRow>
                            )
                          })}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>
          ))}
        </div>

        <UpcomingMatchDetails match={activeMatch} />
      </div>
    </section>
  )
}
