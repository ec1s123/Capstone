import { TrendingDown, TrendingUp } from 'lucide-react'

import { Badge } from '../ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { ClubLogo } from '../shared/ClubLogo'
import { cn } from '../../lib/utils'
import { deltaClass } from '../../lib/formatters'

export function PerformerList({ title, subtitle, items, favoriteTeam, direction }) {
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
