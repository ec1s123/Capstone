// This code was generated with Codex.
import { useMemo, useState } from 'react'
import {
  BarChart3,
  Check,
  Copy,
  Flame,
  Gauge,
  Megaphone,
  MessageSquareQuote,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
} from 'lucide-react'

import { Badge } from '../components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { ClubLogo } from '../components/shared/ClubLogo'
import { SeasonSelector } from '../components/shared/SeasonSelector'
import { cn } from '../lib/utils'
import { deriveMarketPickCode } from '../lib/standings'
import { formatMatchOutcome, formatPercent, formatProbabilityPointGap, formatScoreline } from '../lib/formatters'
import { buildTeamPressureSignals } from '../lib/matchInsights'
import { getDisplayTeamName } from '../lib/teamUtils'

const toneStyles = {
  emerald: {
    card: 'border-emerald-200 bg-emerald-50/45',
    icon: 'border-emerald-200 bg-emerald-100 text-emerald-700',
    badge: 'border-emerald-200 bg-white text-emerald-700',
  },
  sky: {
    card: 'border-sky-200 bg-sky-50/45',
    icon: 'border-sky-200 bg-sky-100 text-sky-700',
    badge: 'border-sky-200 bg-white text-sky-700',
  },
  amber: {
    card: 'border-amber-200 bg-amber-50/55',
    icon: 'border-amber-200 bg-amber-100 text-amber-700',
    badge: 'border-amber-200 bg-white text-amber-700',
  },
  rose: {
    card: 'border-rose-200 bg-rose-50/45',
    icon: 'border-rose-200 bg-rose-100 text-rose-700',
    badge: 'border-rose-200 bg-white text-rose-700',
  },
  slate: {
    card: 'border-slate-200 bg-white',
    icon: 'border-slate-200 bg-slate-100 text-slate-700',
    badge: 'border-slate-200 bg-white text-slate-700',
  },
}

function getOutcomeProbability(match, source, outcomeCode) {
  const prefix = source === 'market' ? 'market' : 'model'
  if (outcomeCode === 'H') return match[`${prefix}HomeProb`]
  if (outcomeCode === 'D') return match[`${prefix}DrawProb`]
  if (outcomeCode === 'A') return match[`${prefix}AwayProb`]
  return null
}

function outcomeName(match, outcomeCode) {
  if (outcomeCode === 'H') return getDisplayTeamName(match.homeTeam)
  if (outcomeCode === 'A') return getDisplayTeamName(match.awayTeam)
  if (outcomeCode === 'D') return 'Draw'
  return 'Unknown'
}

function fixtureName(match) {
  return `${getDisplayTeamName(match.homeTeam)} vs ${getDisplayTeamName(match.awayTeam)}`
}

function formatDisplayMatchOutcome(resultCode, match) {
  if (resultCode === 'H') return `${getDisplayTeamName(match.homeTeam)} Win (H)`
  if (resultCode === 'A') return `${getDisplayTeamName(match.awayTeam)} Win (A)`
  return formatMatchOutcome(resultCode, match)
}

function formatTablePointGap(value) {
  const amount = `${Math.abs(value).toFixed(0)} table points`
  return value >= 0 ? `${amount} above the model` : `${amount} below the model`
}

function isPlayed(match) {
  return Number.isFinite(match.homeGoals) && Number.isFinite(match.awayGoals)
}

function buildCaption({ title, stat, angle, evidence }) {
  const evidenceText = evidence.filter(Boolean).join(' | ')
  return `${title}\n\n${angle}\n\nData: ${stat}${evidenceText ? ` | ${evidenceText}` : ''}`
}

function createTalkingPoints({ matches, topOver, topUnder, currentTable, predictedTable, season }) {
  const points = []
  const playedMatches = matches.filter(isPlayed)
  const pressureSignals = buildTeamPressureSignals(matches)
  const maxPressureSample = Math.max(...pressureSignals.map((team) => team.played), 0)
  const qualifiedPressureSignals = pressureSignals.filter(
    (team) => team.played >= Math.max(3, Math.floor(maxPressureSample * 0.5))
  )
  const pressurePool = qualifiedPressureSignals.length ? qualifiedPressureSignals : pressureSignals

  const edgeMatch = matches
    .map((match) => {
      const modelPick = match.modelPickCode
      const modelProb = getOutcomeProbability(match, 'model', modelPick)
      const marketProb = getOutcomeProbability(match, 'market', modelPick)
      return {
        match,
        modelPick,
        modelProb,
        marketProb,
        edge: modelProb - marketProb,
      }
    })
    .filter((item) => Number.isFinite(item.edge) && item.edge > 0)
    .sort((a, b) => b.edge - a.edge)[0]

  if (edgeMatch) {
    const title = `${outcomeName(edgeMatch.match, edgeMatch.modelPick)} has the largest model-market probability gap.`
    const stat = `${formatPercent(edgeMatch.modelProb)} model probability vs ${formatPercent(edgeMatch.marketProb)} market-implied probability`
    const angle = `${fixtureName(edgeMatch.match)} is the cleanest probability-gap talking point in this season view.`
    points.push({
      id: 'model-edge',
      type: 'Market Gap',
      title,
      angle,
      stat,
      evidence: [`Model rates it ${formatProbabilityPointGap(edgeMatch.edge)}`, edgeMatch.match.matchDate],
      caption: buildCaption({
        title,
        stat,
        angle,
        evidence: [`Model rates it ${formatProbabilityPointGap(edgeMatch.edge)}`],
        season,
      }),
      match: edgeMatch.match,
      tone: 'sky',
      icon: BarChart3,
      score: edgeMatch.edge,
    })
  }

  const disagreement = matches
    .map((match) => {
      const marketPick = deriveMarketPickCode(match)
      const modelProb = getOutcomeProbability(match, 'model', match.modelPickCode)
      const marketProb = getOutcomeProbability(match, 'market', marketPick)
      return {
        match,
        marketPick,
        modelProb,
        marketProb,
        score: Math.abs(modelProb - marketProb) + match.modelConfidence,
      }
    })
    .filter((item) => item.match.modelPickCode !== item.marketPick && Number.isFinite(item.score))
    .sort((a, b) => b.score - a.score)[0]

  if (disagreement) {
    const modelSide = outcomeName(disagreement.match, disagreement.match.modelPickCode)
    const marketSide = outcomeName(disagreement.match, disagreement.marketPick)
    const title = `The model and market split on ${fixtureName(disagreement.match)}.`
    const stat = `Model: ${modelSide} at ${formatPercent(disagreement.modelProb)}. Market lean: ${marketSide} at ${formatPercent(disagreement.marketProb)}.`
    const angle = `That disagreement gives creators a sharper debate angle than a normal match preview.`
    points.push({
      id: 'split-call',
      type: 'Debate Angle',
      title,
      angle,
      stat,
      evidence: [disagreement.match.matchDate],
      caption: buildCaption({ title, stat, angle, evidence: [disagreement.match.matchDate], season }),
      match: disagreement.match,
      tone: 'amber',
      icon: Megaphone,
      score: disagreement.score,
    })
  }

  const strongestPick = [...matches]
    .filter((match) => Number.isFinite(match.modelConfidence))
    .sort((a, b) => b.modelConfidence - a.modelConfidence)[0]

  if (strongestPick) {
    const title = `${outcomeName(strongestPick, strongestPick.modelPickCode)} is the model's strongest call.`
    const stat = `${formatPercent(strongestPick.modelConfidence)} confidence for ${formatDisplayMatchOutcome(strongestPick.modelPickCode, strongestPick)}`
    const angle = `${fixtureName(strongestPick)} is the easiest fixture to frame around model conviction.`
    points.push({
      id: 'strongest-pick',
      type: 'Model Conviction',
      title,
      angle,
      stat,
      evidence: [strongestPick.matchDate],
      caption: buildCaption({ title, stat, angle, evidence: [strongestPick.matchDate], season }),
      match: strongestPick,
      tone: 'emerald',
      icon: Target,
      score: strongestPick.modelConfidence,
    })
  }

  const dominantPressureTeam = pressurePool[0]
  if (dominantPressureTeam) {
    const teamName = getDisplayTeamName(dominantPressureTeam.team)
    const stat = `${dominantPressureTeam.pressurePerMatch.toFixed(1)} shots + corners per match; ${dominantPressureTeam.goalsPerMatch.toFixed(1)} goals per match`
    const title = `${teamName} have the strongest pressure signal.`
    const angle = `Their shot and corner volume makes them the most territorially dominant team in this season view.`
    points.push({
      id: 'pressure-dominance',
      type: 'Pressure Signal',
      title,
      angle,
      stat,
      evidence: [
        `Shot accuracy ${formatPercent(dominantPressureTeam.shotAccuracy)}`,
        `Conversion ${formatPercent(dominantPressureTeam.conversion)}`,
      ],
      caption: buildCaption({
        title,
        stat,
        angle,
        evidence: [
          `Shot accuracy ${formatPercent(dominantPressureTeam.shotAccuracy)}`,
          `Conversion ${formatPercent(dominantPressureTeam.conversion)}`,
        ],
        season,
      }),
      team: dominantPressureTeam.team,
      tone: 'emerald',
      icon: Gauge,
      score: 0.92,
    })
  }

  const lowestPressureTeam = [...pressurePool]
    .sort((a, b) => a.pressurePerMatch - b.pressurePerMatch || a.goalsPerMatch - b.goalsPerMatch)[0]
  if (lowestPressureTeam && lowestPressureTeam.team !== dominantPressureTeam?.team) {
    const teamName = getDisplayTeamName(lowestPressureTeam.team)
    const stat = `${lowestPressureTeam.pressurePerMatch.toFixed(1)} shots + corners per match; ${lowestPressureTeam.goalsPerMatch.toFixed(1)} goals per match`
    const title = `${teamName} have the weakest pressure profile.`
    const angle = `Their low shot and corner volume makes them the clearest pressure concern in this season view.`
    points.push({
      id: 'pressure-concern',
      type: 'Pressure Concern',
      title,
      angle,
      stat,
      evidence: [
        `Shot accuracy ${formatPercent(lowestPressureTeam.shotAccuracy)}`,
        `Conversion ${formatPercent(lowestPressureTeam.conversion)}`,
      ],
      caption: buildCaption({
        title,
        stat,
        angle,
        evidence: [
          `Shot accuracy ${formatPercent(lowestPressureTeam.shotAccuracy)}`,
          `Conversion ${formatPercent(lowestPressureTeam.conversion)}`,
        ],
        season,
      }),
      team: lowestPressureTeam.team,
      tone: 'rose',
      icon: TrendingDown,
      score: 0.9,
    })
  }

  const drawEdge = matches
    .map((match) => ({
      match,
      edge: match.modelDrawProb - match.marketDrawProb,
    }))
    .filter((item) => Number.isFinite(item.edge) && item.edge > 0)
    .sort((a, b) => b.edge - a.edge)[0]

  if (drawEdge) {
    const title = `The draw is more interesting in ${fixtureName(drawEdge.match)} than the market suggests.`
    const stat = `${formatPercent(drawEdge.match.modelDrawProb)} model draw probability vs ${formatPercent(drawEdge.match.marketDrawProb)} market-implied draw probability`
    const angle = `Draw probabilities are easy to overlook, but this match has the largest draw gap in the selected data.`
    points.push({
      id: 'draw-watch',
      type: 'Draw Watch',
      title,
      angle,
      stat,
      evidence: [`Model rates the draw ${formatProbabilityPointGap(drawEdge.edge)}`, drawEdge.match.matchDate],
      caption: buildCaption({
        title,
        stat,
        angle,
        evidence: [`Model rates the draw ${formatProbabilityPointGap(drawEdge.edge)}`],
        season,
      }),
      match: drawEdge.match,
      tone: 'slate',
      icon: MessageSquareQuote,
      score: drawEdge.edge,
    })
  }

  const highConfidenceMiss = [...playedMatches]
    .filter((match) => !match.predictionCorrect && Number.isFinite(match.modelConfidence))
    .sort((a, b) => b.modelConfidence - a.modelConfidence)[0]

  if (highConfidenceMiss) {
    const title = `${fixtureName(highConfidenceMiss)} is the best upset-result talking point.`
    const stat = `Model picked ${formatDisplayMatchOutcome(highConfidenceMiss.modelPickCode, highConfidenceMiss)} at ${formatPercent(highConfidenceMiss.modelConfidence)}, final score ${formatScoreline(highConfidenceMiss.homeGoals, highConfidenceMiss.awayGoals)}`
    const angle = `The result cut against a confident model read, which makes it useful for a hindsight or reaction post.`
    points.push({
      id: 'upset-result',
      type: 'Upset Result',
      title,
      angle,
      stat,
      evidence: [highConfidenceMiss.matchDate],
      caption: buildCaption({ title, stat, angle, evidence: [highConfidenceMiss.matchDate], season }),
      match: highConfidenceMiss,
      tone: 'rose',
      icon: Flame,
      score: highConfidenceMiss.modelConfidence,
    })
  }

  const biggestMargin = [...playedMatches]
    .map((match) => ({
      match,
      margin: Math.abs(match.homeGoals - match.awayGoals),
    }))
    .sort((a, b) => b.margin - a.margin)[0]

  if (biggestMargin?.margin > 0) {
    const title = `${fixtureName(biggestMargin.match)} delivered the biggest scoreline gap.`
    const stat = `Final score ${formatScoreline(biggestMargin.match.homeGoals, biggestMargin.match.awayGoals)}`
    const angle = `This is the cleanest result-based post because the scoreboard did most of the storytelling.`
    points.push({
      id: 'biggest-margin',
      type: 'Result Story',
      title,
      angle,
      stat,
      evidence: [`Margin ${biggestMargin.margin}`, biggestMargin.match.matchDate],
      caption: buildCaption({ title, stat, angle, evidence: [`Margin ${biggestMargin.margin}`], season }),
      match: biggestMargin.match,
      tone: 'emerald',
      icon: Sparkles,
      score: biggestMargin.margin / 10,
    })
  }

  const overPerformer = topOver?.[0]
  if (overPerformer) {
    const gap = overPerformer.points - overPerformer.predictedPoints
    const teamName = getDisplayTeamName(overPerformer.team)
    const title = `${teamName} are beating the model table.`
    const stat = `${overPerformer.points} actual points vs ${overPerformer.predictedPoints} model points`
    const angle = `${formatTablePointGap(gap)} makes them the best over-performance story in the selected season.`
    points.push({
      id: 'over-performer',
      type: 'Table Story',
      title,
      angle,
      stat,
      evidence: [`Actual points are ${formatTablePointGap(gap)}`],
      caption: buildCaption({ title, stat, angle, evidence: [`Actual points are ${formatTablePointGap(gap)}`], season }),
      team: overPerformer.team,
      tone: 'emerald',
      icon: TrendingUp,
      score: Math.max(gap, 0) / 10,
    })
  }

  const underPerformer = topUnder?.[0]
  if (underPerformer) {
    const gap = underPerformer.points - underPerformer.predictedPoints
    const teamName = getDisplayTeamName(underPerformer.team)
    const title = `${teamName} are lagging behind the model table.`
    const stat = `${underPerformer.points} actual points vs ${underPerformer.predictedPoints} model points`
    const angle = `${formatTablePointGap(gap)} turns them into the strongest under-performance talking point.`
    points.push({
      id: 'under-performer',
      type: 'Table Story',
      title,
      angle,
      stat,
      evidence: [`Actual points are ${formatTablePointGap(gap)}`],
      caption: buildCaption({ title, stat, angle, evidence: [`Actual points are ${formatTablePointGap(gap)}`], season }),
      team: underPerformer.team,
      tone: 'rose',
      icon: TrendingDown,
      score: Math.max(Math.abs(gap), 0) / 10,
    })
  }

  const currentLeader = currentTable?.[0]
  const modelLeader = predictedTable?.[0]
  if (currentLeader && modelLeader && currentLeader.team !== modelLeader.team) {
    const title = `The table leader and model leader are not the same.`
    const stat = `${getDisplayTeamName(currentLeader.team)} lead on ${currentLeader.points} actual points; ${getDisplayTeamName(modelLeader.team)} lead the model table on ${modelLeader.predictedPoints} points`
    const angle = `That split is a strong table debate: standings say one thing, the model projection says another.`
    points.push({
      id: 'leader-split',
      type: 'Table Debate',
      title,
      angle,
      stat,
      evidence: [season || 'Selected season'],
      caption: buildCaption({ title, stat, angle, evidence: [], season }),
      teams: [currentLeader.team, modelLeader.team],
      tone: 'amber',
      icon: BarChart3,
      score: 0.7,
    })
  }

  return points.sort((a, b) => b.score - a.score).slice(0, 8)
}

function copyToClipboard(text) {
  if (navigator.clipboard?.writeText) {
    return navigator.clipboard.writeText(text)
  }

  const textArea = document.createElement('textarea')
  textArea.value = text
  textArea.setAttribute('readonly', '')
  textArea.style.position = 'fixed'
  textArea.style.opacity = '0'
  document.body.appendChild(textArea)
  textArea.select()
  document.execCommand('copy')
  document.body.removeChild(textArea)
  return Promise.resolve()
}

function MetricPill({ label, value }) {
  return (
    <div className="rounded-md border border-slate-200 bg-white px-3 py-2">
      <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">{label}</p>
      <p className="mt-1 text-sm font-semibold leading-5 text-slate-900">{value}</p>
    </div>
  )
}

function PointTeams({ point }) {
  if (point.match) {
    return (
      <div className="flex min-w-0 items-center gap-2">
        <ClubLogo team={point.match.homeTeam} size="lg" />
        <div className="min-w-0">
          <p className="text-sm font-semibold text-slate-900">{getDisplayTeamName(point.match.homeTeam)}</p>
          <p className="text-xs text-slate-500">{getDisplayTeamName(point.match.awayTeam)}</p>
        </div>
        <ClubLogo team={point.match.awayTeam} size="lg" />
      </div>
    )
  }

  if (point.team) {
    return (
      <div className="flex min-w-0 items-center gap-2">
        <ClubLogo team={point.team} size="lg" />
        <p className="text-sm font-semibold text-slate-900">{getDisplayTeamName(point.team)}</p>
      </div>
    )
  }

  if (point.teams?.length) {
    return (
      <div className="flex min-w-0 items-center gap-2">
        {point.teams.map((team) => (
          <ClubLogo key={team} team={team} size="lg" />
        ))}
        <p className="text-sm font-semibold text-slate-900">{point.teams.map(getDisplayTeamName).join(' / ')}</p>
      </div>
    )
  }

  return null
}

function TalkingPointCard({ point, copied, onCopy }) {
  const Icon = point.icon
  const tone = toneStyles[point.tone] ?? toneStyles.slate
  const CopyIcon = copied ? Check : Copy

  return (
    <Card className={cn('overflow-hidden border shadow-sm', tone.card)}>
      <CardHeader className="space-y-4 pb-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-3">
            <div className={cn('flex h-10 w-10 shrink-0 items-center justify-center rounded-md border', tone.icon)}>
              <Icon className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <Badge variant="outline" className={cn('uppercase tracking-[0.12em]', tone.badge)}>
                {point.type}
              </Badge>
            </div>
          </div>
          <button
            type="button"
            className={cn(
              'inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md border bg-white text-slate-600 shadow-sm transition-colors hover:border-slate-300 hover:text-slate-900',
              copied && 'border-emerald-200 text-emerald-700'
            )}
            onClick={() => onCopy(point)}
            aria-label={`Copy caption for ${point.type}`}
            title={copied ? 'Copied' : 'Copy caption'}
          >
            <CopyIcon className="h-4 w-4" />
          </button>
        </div>
        <PointTeams point={point} />
        <CardTitle className="text-xl leading-7 text-slate-950">{point.title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm leading-6 text-slate-700">{point.angle}</p>
        <div className="grid gap-2 sm:grid-cols-2">
          <MetricPill label="Data Point" value={point.stat} />
          <MetricPill label="Evidence" value={point.evidence.join(' | ')} />
        </div>
        <div className="rounded-lg border border-slate-200 bg-white/85 p-4">
          <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
            <MessageSquareQuote className="h-3.5 w-3.5" />
            Caption
          </div>
          <p className="whitespace-pre-line text-sm leading-6 text-slate-800">{point.caption}</p>
        </div>
      </CardContent>
    </Card>
  )
}

export function TalkingPointsPage({
  season,
  seasonOptions,
  onSeasonChange,
  matches,
  topOver,
  topUnder,
  currentTable,
  predictedTable,
}) {
  const [copiedId, setCopiedId] = useState('')
  const talkingPoints = useMemo(
    () => createTalkingPoints({ matches, topOver, topUnder, currentTable, predictedTable, season }),
    [currentTable, matches, predictedTable, season, topOver, topUnder]
  )

  const handleCopy = async (point) => {
    await copyToClipboard(point.caption)
    setCopiedId(point.id)
    window.setTimeout(() => setCopiedId(''), 1600)
  }

  return (
    <section className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Talking Points</p>
          <h2 className="max-w-4xl text-3xl font-semibold tracking-tight text-slate-900 md:text-4xl">
            Data-backed angles for match previews, reactions, and social posts.
          </h2>
          <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
            The strongest narratives from model confidence, market disagreement, table gaps, and result surprises.
          </p>
        </div>
        <SeasonSelector season={season} seasonOptions={seasonOptions} onSeasonChange={onSeasonChange} />
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Cards Ready</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{talkingPoints.length}</p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Fixtures Scanned</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{matches.length}</p>
        </div>
        <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Season</p>
          <p className="mt-2 text-2xl font-semibold text-slate-900">{season || '-'}</p>
        </div>
      </div>

      {talkingPoints.length ? (
        <div className="grid gap-4 xl:grid-cols-2">
          {talkingPoints.map((point) => (
            <TalkingPointCard
              key={point.id}
              point={point}
              copied={copiedId === point.id}
              onCopy={handleCopy}
            />
          ))}
        </div>
      ) : (
        <Card className="border-slate-200 bg-white shadow-sm">
          <CardContent className="p-6">
            <p className="text-sm text-slate-600">No talking points are available for this season yet.</p>
          </CardContent>
        </Card>
      )}
    </section>
  )
}
