import { Badge } from '../ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../ui/table'
import { ClubLogo } from '../shared/ClubLogo'
import { FormChips } from '../shared/FormChips'
import { cn } from '../../lib/utils'
import { deltaClass } from '../../lib/formatters'

const gameweeks = Array.from({ length: 38 }, (_, index) => index + 1)

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

export function TableCard({ title, rows, favoriteTeam, mode, gameweek, onGameweekChange }) {
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
