export function deltaClass(delta) {
  if (delta > 0) return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (delta < 0) return 'border-rose-200 bg-rose-50 text-rose-700'
  return 'border-slate-200 bg-slate-100 text-slate-600'
}

export function formResultClass(result) {
  if (result === 'W') return 'bg-emerald-700 text-white'
  if (result === 'L') return 'bg-rose-600 text-white'
  if (result === 'D') return 'bg-slate-500 text-white'
  return 'bg-slate-200 text-slate-400'
}

export const outcomeLabelMap = {
  W: 'Win',
  D: 'Draw',
  L: 'Loss',
}

export function formatMatchOutcome(resultCode, match) {
  if (resultCode === 'H') return `${match.homeTeam} Win (H)`
  if (resultCode === 'A') return `${match.awayTeam} Win (A)`
  if (resultCode === 'D') return 'Draw'
  return 'Unknown'
}

export function outcomeBadgeClass(outcome) {
  if (outcome === 'W') return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (outcome === 'L') return 'border-rose-200 bg-rose-50 text-rose-700'
  return 'border-slate-200 bg-slate-100 text-slate-700'
}

export function confidenceBadgeClass(value) {
  if (value >= 0.6) return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (value >= 0.45) return 'border-amber-200 bg-amber-50 text-amber-700'
  return 'border-slate-200 bg-slate-100 text-slate-700'
}

export function matchOutcomeBadgeClass(resultCode) {
  if (resultCode === 'H') return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (resultCode === 'A') return 'border-rose-200 bg-rose-50 text-rose-700'
  return 'border-slate-200 bg-slate-100 text-slate-700'
}

export function formatScoreline(homeGoals, awayGoals) {
  if (!Number.isFinite(homeGoals) || !Number.isFinite(awayGoals)) return '-'
  return `${homeGoals}-${awayGoals}`
}

export function formatStatPair(homeValue, awayValue) {
  if (!Number.isFinite(homeValue) || !Number.isFinite(awayValue)) return '-'
  return `${homeValue}-${awayValue}`
}

export function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

export function formatOptionalPercent(value) {
  if (!Number.isFinite(value)) return '-'
  return formatPercent(value)
}

export function formatOptionalStat(value, decimals = 0) {
  if (!Number.isFinite(value)) return '-'
  return value.toFixed(decimals)
}

export function formatOdds(value) {
  if (!Number.isFinite(value)) return '-'
  return value.toFixed(2)
}

export function formatSigned(value, decimals = 1) {
  const fixed = value.toFixed(decimals)
  return value > 0 ? `+${fixed}` : fixed
}

export function comparisonDeltaClass(delta, inverse = false) {
  const adjusted = inverse ? -delta : delta
  if (adjusted > 0.05) return 'text-emerald-700'
  if (adjusted < -0.05) return 'text-rose-700'
  return 'text-slate-600'
}
