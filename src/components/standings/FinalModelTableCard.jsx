import { Sparkles } from 'lucide-react'

import { Badge } from '../ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
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
import { normalizeTeamName } from '../../lib/teamUtils'

export function FinalModelTableCard({ rows, favoriteTeam, season }) {
  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-[0.2em] text-muted-foreground">Model Output</p>
            <CardTitle className="text-2xl">Predicted Premier League Table</CardTitle>
            <CardDescription className="mt-2">
              Softmax model projection derived from <code>src/data/prem_odds_predictions_all.csv</code>. Form
              uses W = win, D = draw, and L = loss over the latest five predicted matches. Pts/W/D/L are an
              expectation-fit projection; xPts comes directly from match probabilities.
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="border-slate-200 bg-white text-slate-700">
              {season || 'Season n/a'}
            </Badge>
            <Badge variant="outline" className="gap-1 border-amber-200 bg-amber-50 text-amber-700">
              <Sparkles className="h-3.5 w-3.5" />
              {rows.length} clubs
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pt-0">
        {rows.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Run <code>python src/MLMODEL.py</code> to regenerate <code>src/data/prem_odds_predictions_all.csv</code>.
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
