// This code was generated with Codex.
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
import { deltaClass, comparisonDeltaClass, formatOptionalStat } from '../lib/formatters'
import { getDisplayTeamName, normalizeTeamName } from '../lib/teamUtils'

const gameweeks = Array.from({ length: 38 }, (_, index) => index + 1)

function formatRecord(row) {
  return `${row.won}-${row.drawn}-${row.lost}`
}

function formatGoalsPair(row) {
  return `${row.goalsFor}-${row.goalsAgainst}`
}

function matrixDeltaClass(delta) {
  if (delta > 0) return 'bg-emerald-50 text-emerald-700'
  if (delta < 0) return 'bg-rose-50 text-rose-700'
  return 'bg-slate-50 text-slate-600'
}

function ComparativeTablesCard({ currentTable, predictedTable, favoriteTeam, gameweek, onGameweekChange }) {
  const groupDivider = 'shadow-[inset_0.5px_0_0_rgba(148,163,184,0.22)]'
  const predictedByTeam = new Map(predictedTable.map((row) => [normalizeTeamName(row.team), row]))
  const comparisonRows = currentTable.map((current) => {
    const predicted = predictedByTeam.get(normalizeTeamName(current.team))
    return {
      team: current.team,
      current,
      predicted,
      positionDelta: predicted ? current.position - predicted.position : 0,
      pointsDelta: predicted ? predicted.predictedPoints - current.points : 0,
    }
  })

  return (
    <Card className="overflow-hidden bg-white/85 shadow-sm">
      <CardHeader className="px-4 pb-4 pt-4 sm:px-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Comparison Matrix</p>
            <CardTitle className="text-xl">Current vs predicted standings</CardTitle>
            <CardDescription className="mt-1 text-xs">
              One row per club keeps the live table and forecast directly comparable. Club names use compact Premier League labels to stay readable.
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Select value={String(gameweek)} onValueChange={(value) => onGameweekChange(Number(value))}>
              <SelectTrigger className="h-9 w-[96px] bg-white text-xs text-slate-900">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {gameweeks.map((week) => (
                  <SelectItem key={`comparison-gw-${week}`} value={String(week)}>
                    GW {week}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Badge variant="outline" className="border-sky-200 bg-sky-50 text-sky-700">
              Live vs Forecast
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4 pt-0 sm:px-5">
        <div className="matches-scroll-container overflow-x-auto overflow-y-hidden rounded-lg bg-white/70">
          <Table className="min-w-[1260px] table-fixed border-separate border-spacing-y-1">
            <TableHeader className="bg-transparent [&_tr]:border-b-0">
              <TableRow className="border-b-0 bg-slate-50/70 hover:bg-slate-50/70">
                <TableHead colSpan={2} className="text-center text-slate-700">
                  Club
                </TableHead>
                <TableHead colSpan={5} className={cn('text-center text-sky-700', groupDivider)}>
                  Current
                </TableHead>
                <TableHead colSpan={4} className={cn('text-center text-emerald-700', groupDivider)}>
                  Predicted
                </TableHead>
                <TableHead colSpan={2} className={cn('text-center text-slate-700', groupDivider)}>
                  Difference
                </TableHead>
              </TableRow>
              <TableRow className="border-b-0 bg-slate-50/70 hover:bg-slate-50/70">
                <TableHead className="w-[54px] whitespace-nowrap">Pos</TableHead>
                <TableHead className="w-[176px] whitespace-nowrap">Team</TableHead>
                <TableHead className={cn('w-[64px] whitespace-nowrap text-right', groupDivider)}>Pts</TableHead>
                <TableHead className="w-[86px] whitespace-nowrap text-right">W-D-L</TableHead>
                <TableHead className="w-[76px] whitespace-nowrap text-right">GF-GA</TableHead>
                <TableHead className="w-[56px] whitespace-nowrap text-right">GD</TableHead>
                <TableHead className="w-[112px] whitespace-nowrap">Form</TableHead>
                <TableHead className={cn('w-[64px] whitespace-nowrap text-right', groupDivider)}>Pos</TableHead>
                <TableHead className="w-[64px] whitespace-nowrap text-right">Pred</TableHead>
                <TableHead className="w-[86px] whitespace-nowrap text-right">W-D-L</TableHead>
                <TableHead className="w-[64px] whitespace-nowrap text-right">xPts</TableHead>
                <TableHead className={cn('w-[74px] whitespace-nowrap text-right', groupDivider)}>Pos +/-</TableHead>
                <TableHead className="w-[74px] whitespace-nowrap text-right">Pts +/-</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {comparisonRows.map(({ team, current, predicted, positionDelta, pointsDelta }) => {
                const isFavorite = normalizeTeamName(team) === normalizeTeamName(favoriteTeam)
                const displayTeam = getDisplayTeamName(team)
                const moveLabel = positionDelta > 0 ? `+${positionDelta}` : String(positionDelta)
                const pointsLabel = pointsDelta > 0 ? `+${pointsDelta}` : String(pointsDelta)

                return (
                  <TableRow
                    key={`comparison-${team}`}
                    className={cn(
                      'h-14 border-b-0 bg-white/55 hover:bg-slate-50/70',
                      isFavorite && 'bg-amber-50/70 hover:bg-amber-50'
                    )}
                  >
                    <TableCell className="w-[54px] font-semibold text-sky-700">
                      <span className="inline-flex h-7 w-7 items-center justify-center rounded-full bg-sky-50 text-xs">
                        {current.position}
                      </span>
                    </TableCell>
                    <TableCell className="w-[176px] font-medium">
                      <div className="flex min-w-0 items-center gap-2" title={normalizeTeamName(team)}>
                        <ClubLogo team={team} />
                        <span className="whitespace-nowrap text-sm">{displayTeam}</span>
                      </div>
                    </TableCell>
                    <TableCell className={cn('w-[64px] text-right font-semibold tabular-nums', groupDivider)}>
                      {current.points}
                    </TableCell>
                    <TableCell className="w-[86px] text-right tabular-nums text-slate-700">{formatRecord(current)}</TableCell>
                    <TableCell className="w-[76px] text-right tabular-nums text-slate-700">{formatGoalsPair(current)}</TableCell>
                    <TableCell className={cn('w-[56px] text-right font-semibold tabular-nums', comparisonDeltaClass(current.goalDifference))}>
                      {current.goalDifference > 0 ? `+${current.goalDifference}` : current.goalDifference}
                    </TableCell>
                    <TableCell className="w-[112px]">
                      <FormChips results={current.form} />
                    </TableCell>
                    <TableCell className={cn('w-[64px] text-right font-semibold tabular-nums text-emerald-700', groupDivider)}>
                      {predicted?.position ?? '-'}
                    </TableCell>
                    <TableCell className="w-[64px] text-right font-semibold tabular-nums">{predicted?.predictedPoints ?? '-'}</TableCell>
                    <TableCell className="w-[86px] text-right tabular-nums text-slate-700">
                      {predicted ? formatRecord(predicted) : '-'}
                    </TableCell>
                    <TableCell className="w-[64px] text-right tabular-nums text-slate-600">
                      {predicted ? formatOptionalStat(predicted.expectedPoints, 1) : '-'}
                    </TableCell>
                    <TableCell className={cn('w-[74px] text-right font-semibold tabular-nums', groupDivider, comparisonDeltaClass(positionDelta))}>
                      {predicted ? moveLabel : '-'}
                    </TableCell>
                    <TableCell className={cn('w-[74px] text-right font-semibold tabular-nums', predicted && matrixDeltaClass(pointsDelta))}>
                      <span className={cn('inline-flex min-w-[54px] justify-center rounded-full px-2.5 py-1', predicted && deltaClass(pointsDelta))}>
                        {predicted ? pointsLabel : '-'}
                      </span>
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

      <ComparativeTablesCard
        currentTable={currentTable}
        predictedTable={predictedTable}
        favoriteTeam={favoriteTeam}
        gameweek={gameweek}
        onGameweekChange={onGameweekChange}
      />
    </div>
  )
}
