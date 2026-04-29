import { useEffect, useState } from 'react'
import { Activity, Sparkles, TrendingDown, TrendingUp } from 'lucide-react'

import { Badge } from '../components/ui/badge'
import { Card, CardContent } from '../components/ui/card'
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
import { MatchInsightCard } from '../components/matches/MatchInsightCard'
import { MatchColumnsMenu } from '../components/matches/MatchColumnsMenu'
import { MatchDetailsDrawer } from '../components/matches/MatchDetailsDrawer'
import { cn } from '../lib/utils'
import {
  confidenceBadgeClass,
  formatMatchOutcome,
  formatPercent,
  formatScoreline,
  matchOutcomeBadgeClass,
} from '../lib/formatters'

export function MatchesPage({
  season,
  seasonOptions,
  onSeasonChange,
  matches,
  insights,
  columnVisibility,
  onToggleColumn,
  onResetColumns,
}) {
  const [activeMatchIndex, setActiveMatchIndex] = useState(-1)

  useEffect(() => {
    if (!matches.length) {
      setActiveMatchIndex(-1)
      return
    }
    if (activeMatchIndex >= matches.length) {
      setActiveMatchIndex(matches.length - 1)
    }
  }, [matches, activeMatchIndex])

  useEffect(() => {
    if (activeMatchIndex < 0) return undefined
    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        setActiveMatchIndex(-1)
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [activeMatchIndex])

  const isVisible = (columnKey) => columnVisibility[columnKey] !== false

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Matches & Results</p>
          <h2 className="text-3xl font-semibold tracking-tight text-slate-900">All Fixtures, Scores, and Confidence</h2>
          <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
            Review every played fixture for the selected season with model confidence, probabilities, and key match stats.
          </p>
        </div>
        <SeasonSelector season={season} seasonOptions={seasonOptions} onSeasonChange={onSeasonChange} />
      </div>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MatchInsightCard
          title="Highest Confidence"
          description="Model's strongest pick"
          match={insights.highestConfidence}
          tone="positive"
          icon={TrendingUp}
        />
        <MatchInsightCard
          title="Lowest Confidence"
          description="Closest-call fixture"
          match={insights.lowestConfidence}
          tone="caution"
          icon={Activity}
        />
        <MatchInsightCard
          title="High-Confidence Miss"
          description="Biggest confident miss"
          match={insights.highestConfidenceMiss}
          tone="negative"
          icon={TrendingDown}
        />
        <MatchInsightCard
          title="Biggest Winning Margin"
          description="Largest scoreline gap"
          match={insights.biggestGoalMargin}
          tone="highlight"
          icon={Sparkles}
        />
      </section>

      <Card className="border-slate-200 bg-white shadow-sm">
        <CardContent className="pt-6">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <p className="text-sm text-slate-600">{matches.length} fixtures</p>
              <p className="text-xs text-slate-500">Click any row to open the side detail panel.</p>
            </div>
            <MatchColumnsMenu
              columnVisibility={columnVisibility}
              onToggleColumn={onToggleColumn}
              onResetColumns={onResetColumns}
            />
          </div>

          <div className="matches-scroll-container overflow-x-auto overflow-y-hidden rounded-lg border border-slate-200">
            <Table className="min-w-[980px]">
              <TableHeader className="bg-slate-50">
                <TableRow className="border-b-0 hover:bg-transparent">
                  {isVisible('date') && <TableHead>Date</TableHead>}
                  {isVisible('home') && <TableHead>Home</TableHead>}
                  {isVisible('away') && <TableHead>Away</TableHead>}
                  {isVisible('score') && <TableHead className="text-right">Score</TableHead>}
                  {isVisible('result') && <TableHead>Result</TableHead>}
                  {isVisible('modelPick') && <TableHead>Model Pick</TableHead>}
                  {isVisible('confidence') && <TableHead>Confidence</TableHead>}
                  {isVisible('prediction') && <TableHead>Prediction</TableHead>}
                </TableRow>
              </TableHeader>
              <TableBody>
                {matches.map((match, index) => (
                  <TableRow
                    key={match.id}
                    className="cursor-pointer hover:bg-slate-50"
                    onClick={() => setActiveMatchIndex(index)}
                  >
                    {isVisible('date') && <TableCell className="font-medium tabular-nums">{match.matchDate}</TableCell>}
                    {isVisible('home') && (
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <ClubLogo team={match.homeTeam} />
                          <span>{match.homeTeam}</span>
                        </div>
                      </TableCell>
                    )}
                    {isVisible('away') && (
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <ClubLogo team={match.awayTeam} />
                          <span>{match.awayTeam}</span>
                        </div>
                      </TableCell>
                    )}
                    {isVisible('score') && (
                      <TableCell className="text-right font-semibold tabular-nums">
                        {formatScoreline(match.homeGoals, match.awayGoals)}
                      </TableCell>
                    )}
                    {isVisible('result') && (
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={cn('font-semibold tracking-normal', matchOutcomeBadgeClass(match.fullTimeResult))}
                        >
                          {formatMatchOutcome(match.fullTimeResult, match)}
                        </Badge>
                      </TableCell>
                    )}
                    {isVisible('modelPick') && (
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={cn('font-semibold tracking-normal', matchOutcomeBadgeClass(match.modelPickCode))}
                        >
                          {formatMatchOutcome(match.modelPickCode, match)}
                        </Badge>
                      </TableCell>
                    )}
                    {isVisible('confidence') && (
                      <TableCell>
                        <span
                          className={cn(
                            'inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold tabular-nums',
                            confidenceBadgeClass(match.modelConfidence)
                          )}
                        >
                          {formatPercent(match.modelConfidence)}
                        </span>
                      </TableCell>
                    )}
                    {isVisible('prediction') && (
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={cn(
                            'uppercase tracking-[0.12em]',
                            match.predictionCorrect
                              ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                              : 'border-rose-200 bg-rose-50 text-rose-700'
                          )}
                        >
                          {match.predictionCorrect ? 'Correct' : 'Miss'}
                        </Badge>
                      </TableCell>
                    )}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <MatchDetailsDrawer
        matches={matches}
        activeIndex={activeMatchIndex}
        onClose={() => setActiveMatchIndex(-1)}
        onSelectIndex={setActiveMatchIndex}
      />
    </section>
  )
}
