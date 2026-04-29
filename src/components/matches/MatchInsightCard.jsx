import { Activity } from 'lucide-react'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { cn } from '../../lib/utils'
import { formatMatchOutcome, formatPercent, formatScoreline } from '../../lib/formatters'

export function MatchInsightCard({ title, description, match, tone = 'neutral', icon: Icon = Activity }) {
  const toneStyles = {
    positive: {
      card: 'border-emerald-200 bg-emerald-50/35',
      title: 'text-emerald-900',
      description: 'text-emerald-800/90',
      iconContainer: 'border-emerald-200 bg-emerald-100 text-emerald-700',
      model: 'text-emerald-800',
    },
    caution: {
      card: 'border-amber-200 bg-amber-50/35',
      title: 'text-amber-900',
      description: 'text-amber-800/90',
      iconContainer: 'border-amber-200 bg-amber-100 text-amber-700',
      model: 'text-amber-800',
    },
    negative: {
      card: 'border-rose-200 bg-rose-50/35',
      title: 'text-rose-900',
      description: 'text-rose-800/90',
      iconContainer: 'border-rose-200 bg-rose-100 text-rose-700',
      model: 'text-rose-800',
    },
    highlight: {
      card: 'border-sky-200 bg-sky-50/35',
      title: 'text-sky-900',
      description: 'text-sky-800/90',
      iconContainer: 'border-sky-200 bg-sky-100 text-sky-700',
      model: 'text-sky-800',
    },
    neutral: {
      card: 'border-slate-200 bg-white',
      title: 'text-slate-900',
      description: 'text-slate-600',
      iconContainer: 'border-slate-200 bg-slate-100 text-slate-600',
      model: 'text-slate-700',
    },
  }
  const palette = toneStyles[tone] ?? toneStyles.neutral

  return (
    <Card className={cn('shadow-sm', palette.card)}>
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className={cn('text-base', palette.title)}>{title}</CardTitle>
            <CardDescription className={palette.description}>{description}</CardDescription>
          </div>
          <div className={cn('flex h-8 w-8 shrink-0 items-center justify-center rounded-full border', palette.iconContainer)}>
            <Icon className="h-4 w-4" />
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-1 pt-0">
        {match ? (
          <>
            <p className="text-sm font-semibold text-slate-900">
              {match.homeTeam} {formatScoreline(match.homeGoals, match.awayGoals)} {match.awayTeam}
            </p>
            <p className="text-xs text-muted-foreground">{match.matchDate}</p>
            <p className={cn('text-xs', palette.model)}>
              Model: {formatMatchOutcome(match.modelPickCode, match)} ({formatPercent(match.modelConfidence)})
            </p>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">No matching fixtures available.</p>
        )}
      </CardContent>
    </Card>
  )
}
