import { useEffect, useState } from 'react'

import { Badge } from '../components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
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
import { SeasonSelector } from '../components/shared/SeasonSelector'
import { MatchDetailsDrawer } from '../components/matches/MatchDetailsDrawer'
import { cn } from '../lib/utils'
import {
  comparisonDeltaClass,
  confidenceBadgeClass,
  formatPercent,
  formatSigned,
  outcomeBadgeClass,
  outcomeLabelMap,
} from '../lib/formatters'

export function ClubPage({
  season,
  seasonOptions,
  onSeasonChange,
  clubs,
  selectedClub,
  onSelectedClubChange,
  clubFixtures,
  clubSummary,
}) {
  const [activeMatchIndex, setActiveMatchIndex] = useState(-1)

  useEffect(() => {
    if (!clubFixtures.length) {
      setActiveMatchIndex(-1)
      return
    }
    if (activeMatchIndex >= clubFixtures.length) {
      setActiveMatchIndex(clubFixtures.length - 1)
    }
  }, [clubFixtures, activeMatchIndex])

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Club View</p>
          <h2 className="text-3xl font-semibold tracking-tight text-slate-900">
            Fixtures, Results, and Model Confidence
          </h2>
          <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
            Select any club to view all fixtures, match outcomes, and model confidence for the chosen season.
          </p>
        </div>
        <SeasonSelector season={season} seasonOptions={seasonOptions} onSeasonChange={onSeasonChange} />
      </div>

      <Card className="border-slate-200 bg-white shadow-sm">
        <CardHeader className="space-y-5 pb-4">
          <div>
            <CardTitle className="text-lg">Club Breakdown</CardTitle>
          </div>

          <div className="space-y-2">
            <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Choose Club</p>
            <Select value={selectedClub} onValueChange={onSelectedClubChange}>
              <SelectTrigger className="h-16 w-full border border-slate-300 bg-white px-5 text-lg font-semibold text-slate-900 sm:h-20 sm:text-2xl">
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

              <div>
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
            <p className="text-sm text-muted-foreground">No fixtures available for this club in {season || 'this season'}.</p>
          )}

          <div className="overflow-x-auto overflow-y-hidden rounded-lg border border-slate-200">
            <div className="border-b border-slate-200 bg-slate-50 px-4 py-2">
              <p className="text-xs text-slate-500">Click any row to open the side detail panel.</p>
            </div>
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
                {clubFixtures.map((fixture, index) => (
                  <TableRow
                    key={fixture.id}
                    className="h-14 cursor-pointer hover:bg-slate-50"
                    onClick={() => setActiveMatchIndex(index)}
                  >
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

      <MatchDetailsDrawer
        matches={clubFixtures}
        activeIndex={activeMatchIndex}
        onClose={() => setActiveMatchIndex(-1)}
        onSelectIndex={setActiveMatchIndex}
      />
    </section>
  )
}
