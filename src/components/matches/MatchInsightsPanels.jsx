import { ClubLogo } from '../shared/ClubLogo'
import { cn } from '../../lib/utils'
import {
  comparisonDeltaClass,
  formatOptionalPercent,
  formatPercent,
  formatSigned,
} from '../../lib/formatters'
import { barPercent, probabilityBarHeight, safeChartValue } from '../../lib/matchInsights'

export function MatchSummaryTiles({ summary }) {
  return (
    <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
      {summary.map((item) => (
        <div key={item.label} className="rounded-lg border border-slate-200 bg-white p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{item.label}</p>
          <p className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">{item.value}</p>
          <p className="mt-1 text-sm text-slate-600">{item.detail}</p>
        </div>
      ))}
    </section>
  )
}

export function OutcomeMixChart({ title, rows }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{title}</p>
      <div className="mt-4 space-y-3">
        {rows.map((row) => (
          <div key={`${title}-${row.code}`} className="grid grid-cols-[5.5rem_1fr_3.5rem] items-center gap-3 text-sm">
            <span className="font-medium text-slate-700">{row.label}</span>
            <div className="h-2 overflow-hidden rounded-full bg-slate-100">
              <div className="h-full rounded-full bg-slate-900" style={{ width: probabilityBarHeight(row.share) }} />
            </div>
            <span className="text-right tabular-nums text-slate-600">{formatPercent(row.share)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function GameTexturePanel({ texture }) {
  const maxValue = Math.max(...texture.map((item) => (item.format === 'percent' ? safeChartValue(item.value) * 100 : safeChartValue(item.value))), 1)

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Game Texture</p>
      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        {texture.map((item) => {
          const displayValue =
            item.format === 'percent'
              ? formatOptionalPercent(item.value)
              : Number.isFinite(item.value)
                ? item.value.toFixed(1)
                : '-'
          const scaledValue = item.format === 'percent' ? safeChartValue(item.value) * 100 : safeChartValue(item.value)

          return (
            <div key={item.label} className="space-y-2">
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="font-medium text-slate-700">{item.label}</span>
                <span className="tabular-nums text-slate-900">{displayValue}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                <div className="h-full rounded-full bg-sky-500" style={{ width: barPercent(scaledValue, maxValue) }} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function TeamSignalPanel({ teams }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Team Pressure Signals</p>
        <p className="text-xs text-slate-500">Top shot + corner volume</p>
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[680px] text-left text-sm">
          <thead className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
            <tr className="border-b border-slate-200">
              <th className="py-2 pr-3">Team</th>
              <th className="px-3 py-2 text-right">Pressure</th>
              <th className="px-3 py-2 text-right">Goals</th>
              <th className="px-3 py-2 text-right">Shot Acc.</th>
              <th className="px-3 py-2 text-right">Conversion</th>
              <th className="pl-3 py-2 text-right">Pts vs xPts</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200">
            {teams.map((team) => (
              <tr key={`team-signal-${team.team}`}>
                <td className="py-3 pr-3">
                  <div className="flex items-center gap-2">
                    <ClubLogo team={team.team} />
                    <span className="font-semibold text-slate-900">{team.team}</span>
                  </div>
                </td>
                <td className="px-3 py-3 text-right tabular-nums text-slate-700">{team.pressurePerMatch.toFixed(1)}</td>
                <td className="px-3 py-3 text-right tabular-nums text-slate-700">{team.goalsPerMatch.toFixed(1)}</td>
                <td className="px-3 py-3 text-right tabular-nums text-slate-700">{formatPercent(team.shotAccuracy)}</td>
                <td className="px-3 py-3 text-right tabular-nums text-slate-700">{formatPercent(team.conversion)}</td>
                <td className={cn('pl-3 py-3 text-right tabular-nums font-semibold', comparisonDeltaClass(team.pointDelta))}>
                  {formatSigned(team.pointDelta, 1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export function ModelEdgePanel({ matches, onSelectMatch }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Model Edges vs Market</p>
      <div className="mt-4 space-y-3">
        {matches.map((match) => (
          <button
            key={`model-edge-${match.id}`}
            type="button"
            className="grid w-full grid-cols-[1fr_auto] items-center gap-3 rounded-md px-2 py-2 text-left hover:bg-slate-50"
            onClick={() => onSelectMatch(match)}
          >
            <span className="min-w-0">
              <span className="block truncate text-sm font-semibold text-slate-900">
                {match.homeTeam} vs {match.awayTeam}
              </span>
              <span className="block text-xs text-slate-500">
                {match.edgeLabel} edge · {match.matchDate}
              </span>
            </span>
            <span className={cn('tabular-nums text-sm font-semibold', comparisonDeltaClass(match.edgeDelta))}>
              {formatSigned(match.edgeDelta * 100, 1)} pts
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
