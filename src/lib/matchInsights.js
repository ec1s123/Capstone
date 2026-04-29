import { deriveMarketPickCode, outcomeForClub } from './standings'
import { formatMatchOutcome, formatPercent } from './formatters'

export function average(values) {
  const finiteValues = values.filter(Number.isFinite)
  if (!finiteValues.length) return null
  return finiteValues.reduce((sum, value) => sum + value, 0) / finiteValues.length
}

export function sumFinite(values) {
  return values.reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0)
}

export function buildOutcomeMix(matches, pickKey = 'fullTimeResult') {
  const total = Math.max(matches.length, 1)
  return [
    { code: 'H', label: 'Home Team Wins', count: matches.filter((match) => match[pickKey] === 'H').length },
    { code: 'D', label: 'Draw', count: matches.filter((match) => match[pickKey] === 'D').length },
    { code: 'A', label: 'Away Team Wins', count: matches.filter((match) => match[pickKey] === 'A').length },
  ].map((row) => ({ ...row, share: row.count / total }))
}

export function buildMatchPageInsightData(matches) {
  if (!matches.length) {
    return {
      summary: [],
      actualOutcomeMix: [],
      modelOutcomeMix: [],
      marketOutcomeMix: [],
      gameTexture: [],
      teamSignals: [],
      modelEdges: [],
    }
  }

  const playedMatches = matches.filter((match) => Number.isFinite(match.homeGoals) && Number.isFinite(match.awayGoals))
  const modelHits = matches.filter((match) => match.predictionCorrect).length
  const marketHits = matches.filter((match) => deriveMarketPickCode(match) === match.fullTimeResult).length
  const modelMarketAgreements = matches.filter((match) => deriveMarketPickCode(match) === match.modelPickCode).length
  const highConfidenceMatches = matches.filter((match) => match.modelConfidence >= 0.6)
  const highConfidenceHits = highConfidenceMatches.filter((match) => match.predictionCorrect).length
  const goalTotals = playedMatches.map((match) => safeChartValue(match.homeGoals) + safeChartValue(match.awayGoals))
  const shotTotals = matches.map((match) => safeChartValue(match.homeShots) + safeChartValue(match.awayShots))
  const shotOnTargetTotals = matches.map((match) => safeChartValue(match.homeShotsOnTarget) + safeChartValue(match.awayShotsOnTarget))
  const cornerTotals = matches.map((match) => safeChartValue(match.homeCorners) + safeChartValue(match.awayCorners))
  const cardTotals = matches.map(
    (match) =>
      safeChartValue(match.homeYellowCards) +
      safeChartValue(match.awayYellowCards) +
      safeChartValue(match.homeRedCards) * 2 +
      safeChartValue(match.awayRedCards) * 2
  )
  const totalShots = sumFinite(matches.flatMap((match) => [match.homeShots, match.awayShots]))
  const totalShotsOnTarget = sumFinite(matches.flatMap((match) => [match.homeShotsOnTarget, match.awayShotsOnTarget]))
  const overTwoPointFiveGoals = playedMatches.filter(
    (match) => safeChartValue(match.homeGoals) + safeChartValue(match.awayGoals) >= 3
  ).length

  const teamMap = new Map()
  const ensureTeam = (team) => {
    if (!teamMap.has(team)) {
      teamMap.set(team, {
        team,
        played: 0,
        actualPoints: 0,
        expectedPoints: 0,
        goalsFor: 0,
        goalsAgainst: 0,
        shots: 0,
        shotsOnTarget: 0,
        corners: 0,
        cards: 0,
      })
    }
    return teamMap.get(team)
  }

  matches.forEach((match) => {
    const home = ensureTeam(match.homeTeam)
    const away = ensureTeam(match.awayTeam)

    home.played += 1
    away.played += 1

    home.goalsFor += safeChartValue(match.homeGoals)
    home.goalsAgainst += safeChartValue(match.awayGoals)
    away.goalsFor += safeChartValue(match.awayGoals)
    away.goalsAgainst += safeChartValue(match.homeGoals)

    home.shots += safeChartValue(match.homeShots)
    away.shots += safeChartValue(match.awayShots)
    home.shotsOnTarget += safeChartValue(match.homeShotsOnTarget)
    away.shotsOnTarget += safeChartValue(match.awayShotsOnTarget)
    home.corners += safeChartValue(match.homeCorners)
    away.corners += safeChartValue(match.awayCorners)
    home.cards += safeChartValue(match.homeYellowCards) + safeChartValue(match.homeRedCards) * 2
    away.cards += safeChartValue(match.awayYellowCards) + safeChartValue(match.awayRedCards) * 2

    home.expectedPoints += match.modelHomeProb * 3 + match.modelDrawProb
    away.expectedPoints += match.modelAwayProb * 3 + match.modelDrawProb

    const homeOutcome = outcomeForClub(match.fullTimeResult, true)
    const awayOutcome = outcomeForClub(match.fullTimeResult, false)
    if (homeOutcome === 'W') home.actualPoints += 3
    if (homeOutcome === 'D') home.actualPoints += 1
    if (awayOutcome === 'W') away.actualPoints += 3
    if (awayOutcome === 'D') away.actualPoints += 1
  })

  const teamSignals = [...teamMap.values()]
    .map((team) => ({
      ...team,
      goalsPerMatch: team.played ? team.goalsFor / team.played : 0,
      shotsPerMatch: team.played ? team.shots / team.played : 0,
      pressurePerMatch: team.played ? (team.shots + team.corners) / team.played : 0,
      shotAccuracy: team.shots ? team.shotsOnTarget / team.shots : 0,
      conversion: team.shots ? team.goalsFor / team.shots : 0,
      pointDelta: team.actualPoints - team.expectedPoints,
    }))
    .sort((a, b) => b.pressurePerMatch - a.pressurePerMatch || b.goalsPerMatch - a.goalsPerMatch)
    .slice(0, 6)

  const modelEdges = matches
    .map((match) => {
      const edges = outcomeProbabilityRows(match).map((row) => ({
        label: row.label,
        delta: row.model - row.market,
      }))
      const strongestEdge = edges.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))[0]
      return { ...match, edgeLabel: strongestEdge.label, edgeDelta: strongestEdge.delta }
    })
    .sort((a, b) => Math.abs(b.edgeDelta) - Math.abs(a.edgeDelta))
    .slice(0, 5)

  return {
    summary: [
      {
        label: 'Model Accuracy',
        value: formatPercent(modelHits / matches.length),
        detail: `${modelHits}/${matches.length} correct`,
      },
      {
        label: 'Market Accuracy',
        value: formatPercent(marketHits / matches.length),
        detail: `${marketHits}/${matches.length} correct`,
      },
      {
        label: 'Model-Market Agreement',
        value: formatPercent(modelMarketAgreements / matches.length),
        detail: `${modelMarketAgreements} aligned picks`,
      },
      {
        label: 'High-Confidence Hit Rate',
        value: highConfidenceMatches.length ? formatPercent(highConfidenceHits / highConfidenceMatches.length) : '-',
        detail: `${highConfidenceMatches.length} picks at 60%+`,
      },
    ],
    actualOutcomeMix: buildOutcomeMix(matches, 'fullTimeResult'),
    modelOutcomeMix: buildOutcomeMix(matches, 'modelPickCode'),
    marketOutcomeMix: buildOutcomeMix(matches.map((match) => ({ ...match, marketPickCode: deriveMarketPickCode(match) })), 'marketPickCode'),
    gameTexture: [
      { label: 'Goals / Match', value: average(goalTotals), format: 'number' },
      { label: 'Over 2.5 Goals', value: playedMatches.length ? overTwoPointFiveGoals / playedMatches.length : null, format: 'percent' },
      { label: 'Shots / Match', value: average(shotTotals), format: 'number' },
      { label: 'Shot Accuracy', value: totalShots ? totalShotsOnTarget / totalShots : null, format: 'percent' },
      { label: 'Corners / Match', value: average(cornerTotals), format: 'number' },
      { label: 'Card Load / Match', value: average(cardTotals), format: 'number' },
    ],
    teamSignals,
    modelEdges,
  }
}

export function safeChartValue(value) {
  return Number.isFinite(value) ? value : 0
}

export function ratioOrNull(part, whole) {
  if (!Number.isFinite(part) || !Number.isFinite(whole) || whole <= 0) return null
  return part / whole
}

export function barPercent(value, maxValue = 1) {
  const safeValue = Math.max(0, safeChartValue(value))
  const safeMax = Math.max(1, safeChartValue(maxValue))
  if (safeValue <= 0) return '0%'
  return `${Math.max((safeValue / safeMax) * 100, 3)}%`
}

export function probabilityBarHeight(value) {
  const safeValue = Math.max(0, Math.min(1, safeChartValue(value)))
  if (safeValue <= 0) return '0%'
  return `${Math.max(safeValue * 100, 3)}%`
}

export function outcomeProbabilityRows(match) {
  return [
    {
      code: 'H',
      label: formatMatchOutcome('H', match),
      shortLabel: match.homeTeam,
      team: match.homeTeam,
      model: match.modelHomeProb,
      market: match.marketHomeProb,
      odds: match.b365HomeOdds,
    },
    {
      code: 'D',
      label: 'Draw',
      shortLabel: 'Draw',
      team: 'Draw',
      model: match.modelDrawProb,
      market: match.marketDrawProb,
      odds: match.b365DrawOdds,
    },
    {
      code: 'A',
      label: formatMatchOutcome('A', match),
      shortLabel: match.awayTeam,
      team: match.awayTeam,
      model: match.modelAwayProb,
      market: match.marketAwayProb,
      odds: match.b365AwayOdds,
    },
  ]
}
