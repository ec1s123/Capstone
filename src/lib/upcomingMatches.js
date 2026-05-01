// This code was generated with Codex.
import { normalizeTeamName } from './teamUtils'
import {
  buildActualTable,
  buildModelOutputTable,
  derivePickCodeFromProbabilities,
  emptyForm,
  sortFixturesChronologically,
} from './standings'

function average(values, fallback = 0) {
  const finiteValues = values.filter(Number.isFinite)
  if (!finiteValues.length) return fallback
  return finiteValues.reduce((sum, value) => sum + value, 0) / finiteValues.length
}

function formPointsPerMatch(form) {
  const validResults = form.filter(Boolean)
  if (!validResults.length) return 1
  const points = validResults.reduce((sum, result) => {
    if (result === 'W') return sum + 3
    if (result === 'D') return sum + 1
    return sum
  }, 0)
  return points / validResults.length
}

function softmax(logits) {
  const maxLogit = Math.max(...logits)
  const exponentials = logits.map((logit) => Math.exp(logit - maxLogit))
  const total = exponentials.reduce((sum, value) => sum + value, 0)
  return exponentials.map((value) => value / total)
}

function roundTo(value, decimals = 2) {
  return Number(value.toFixed(decimals))
}

function scheduleSortValue(match) {
  return `${match.matchDate} ${match.kickoffTime ?? '99:99'} ${match.homeTeam} ${match.awayTeam}`
}

function resultKey(team) {
  return normalizeTeamName(team)
}

function buildTeamProfiles(seasonResults, teams) {
  const actualRows = buildActualTable(seasonResults, teams)
  const modelRows = buildModelOutputTable(seasonResults)
  const actualByTeam = new Map(
    actualRows.map((row, index) => [resultKey(row.team), { ...row, position: index + 1 }])
  )
  const modelByTeam = new Map(
    modelRows.map((row, index) => [resultKey(row.Team), { ...row, position: index + 1 }])
  )

  const profiles = new Map()

  teams.forEach((team) => {
    const normalizedTeam = normalizeTeamName(team)
    const actual = actualByTeam.get(normalizedTeam)
    const model = modelByTeam.get(normalizedTeam)
    const played = actual?.played ?? model?.Played ?? 0
    const form = actual?.form ?? emptyForm()
    const predictedForm = model?.Form ?? emptyForm()

    profiles.set(normalizedTeam, {
      team: normalizedTeam,
      played,
      actualPosition: actual?.position ?? null,
      actualPoints: actual?.points ?? 0,
      pointsPerMatch: played ? (actual?.points ?? 0) / played : 1,
      goalDifference: actual?.goalDifference ?? 0,
      goalDifferencePerMatch: played ? (actual?.goalDifference ?? 0) / played : 0,
      goalsForPerMatch: played ? (actual?.goalsFor ?? 0) / played : 1.3,
      goalsAgainstPerMatch: played ? (actual?.goalsAgainst ?? 0) / played : 1.3,
      modelPosition: model?.position ?? null,
      modelPoints: model?.Points ?? 0,
      expectedPoints: model?.ExpectedPoints ?? 0,
      expectedPointsPerMatch: played ? (model?.ExpectedPoints ?? 0) / played : 1,
      form,
      predictedForm,
      formPointsPerMatch: formPointsPerMatch(form),
    })
  })

  return profiles
}

function profileAverages(profiles) {
  const rows = [...profiles.values()]
  return {
    pointsPerMatch: average(rows.map((row) => row.pointsPerMatch), 1.35),
    expectedPointsPerMatch: average(rows.map((row) => row.expectedPointsPerMatch), 1.35),
    formPointsPerMatch: average(rows.map((row) => row.formPointsPerMatch), 1.35),
  }
}

function teamSignal(profile, averages) {
  return (
    (profile.expectedPointsPerMatch - averages.expectedPointsPerMatch) * 0.85 +
    (profile.pointsPerMatch - averages.pointsPerMatch) * 0.35 +
    (profile.formPointsPerMatch - averages.formPointsPerMatch) * 0.22 +
    profile.goalDifferencePerMatch * 0.1
  )
}

function projectMatchProbabilities(homeProfile, awayProfile, averages) {
  const matchupSignal = teamSignal(homeProfile, averages) - teamSignal(awayProfile, averages)
  const closenessPenalty = Math.abs(matchupSignal) * 0.3
  const [home, draw, away] = softmax([
    0.18 + matchupSignal,
    -0.38 - closenessPenalty,
    -0.04 - matchupSignal,
  ])

  return {
    home,
    draw,
    away,
    matchupSignal,
  }
}

function teamPositionText(profile) {
  if (!Number.isFinite(profile.actualPosition)) return 'Unranked'
  return `${profile.actualPosition}${ordinalSuffix(profile.actualPosition)}`
}

function ordinalSuffix(value) {
  const lastTwo = value % 100
  if (lastTwo >= 11 && lastTwo <= 13) return 'th'
  const last = value % 10
  if (last === 1) return 'st'
  if (last === 2) return 'nd'
  if (last === 3) return 'rd'
  return 'th'
}

function buildSignalLabel(match) {
  if (match.modelConfidence < 0.42) return 'Toss-up'
  if (Math.abs(match.tablePositionGap) >= 8) return 'Table gap'
  if (Math.abs(match.formPointsGap) >= 0.7) return 'Form edge'
  return 'Model lean'
}

function buildNotes(match) {
  const notes = []
  const pickTeam =
    match.modelPickCode === 'H' ? match.homeTeam : match.modelPickCode === 'A' ? match.awayTeam : 'Draw'

  if (match.modelConfidence >= 0.55) {
    notes.push(`${pickTeam} is the clearest model lean at ${Math.round(match.modelConfidence * 100)}%.`)
  } else if (match.modelConfidence < 0.42) {
    notes.push('The model sees a narrow split across the three outcomes.')
  } else {
    notes.push(`${pickTeam} has the edge, but the confidence band is moderate.`)
  }

  if (Math.abs(match.expectedPointsGap) >= 0.45) {
    const advantagedTeam = match.expectedPointsGap > 0 ? match.homeTeam : match.awayTeam
    notes.push(`${advantagedTeam} owns the stronger expected-points projection for this fixture.`)
  }

  if (Math.abs(match.formPointsGap) >= 0.7) {
    const formTeam = match.formPointsGap > 0 ? match.homeTeam : match.awayTeam
    notes.push(`${formTeam} has the better recent results profile over the last five available matches.`)
  }

  const teamsInBottomFight = [match.homeProfile, match.awayProfile].filter(
    (profile) => Number.isFinite(profile.actualPosition) && profile.actualPosition >= 16
  )
  if (teamsInBottomFight.length) {
    notes.push(`${teamsInBottomFight.map((profile) => profile.team).join(' and ')} sit in the survival-pressure band.`)
  }

  const teamsInEuropeanRace = [match.homeProfile, match.awayProfile].filter(
    (profile) => Number.isFinite(profile.actualPosition) && profile.actualPosition <= 7
  )
  if (teamsInEuropeanRace.length) {
    notes.push(`${teamsInEuropeanRace.map((profile) => profile.team).join(' and ')} are carrying top-seven table stakes.`)
  }

  if (match.dateNote) notes.push(match.dateNote)

  return notes.slice(0, 4)
}

export function sortUpcomingMatches(matches) {
  return [...matches].sort((a, b) => scheduleSortValue(a).localeCompare(scheduleSortValue(b)))
}

export function buildUpcomingMatches(fixtures, seasonResults, seasonTeams) {
  const fixtureTeams = fixtures.flatMap((fixture) => [
    normalizeTeamName(fixture.homeTeam),
    normalizeTeamName(fixture.awayTeam),
  ])
  const teams = [...new Set([...seasonTeams.map(normalizeTeamName), ...fixtureTeams])]
  const profiles = buildTeamProfiles(sortFixturesChronologically(seasonResults), teams)
  const averages = profileAverages(profiles)

  return sortUpcomingMatches(
    fixtures.map((fixture) => {
      const homeTeam = normalizeTeamName(fixture.homeTeam)
      const awayTeam = normalizeTeamName(fixture.awayTeam)
      const homeProfile = profiles.get(homeTeam)
      const awayProfile = profiles.get(awayTeam)
      const probabilities = projectMatchProbabilities(homeProfile, awayProfile, averages)
      const modelPickCode = derivePickCodeFromProbabilities(probabilities.home, probabilities.draw, probabilities.away)
      const modelConfidence = Math.max(probabilities.home, probabilities.draw, probabilities.away)
      const homeExpectedPoints = probabilities.home * 3 + probabilities.draw
      const awayExpectedPoints = probabilities.away * 3 + probabilities.draw
      const enrichedMatch = {
        ...fixture,
        roundLabel: fixture.roundLabel ?? `Gameweek ${fixture.gameweek}`,
        homeTeam,
        awayTeam,
        homeProfile,
        awayProfile,
        modelPickCode,
        modelConfidence,
        modelHomeProb: probabilities.home,
        modelDrawProb: probabilities.draw,
        modelAwayProb: probabilities.away,
        homeExpectedPoints: roundTo(homeExpectedPoints),
        awayExpectedPoints: roundTo(awayExpectedPoints),
        expectedPointsGap: homeExpectedPoints - awayExpectedPoints,
        formPointsGap: homeProfile.formPointsPerMatch - awayProfile.formPointsPerMatch,
        tablePositionGap:
          Number.isFinite(homeProfile.actualPosition) && Number.isFinite(awayProfile.actualPosition)
            ? homeProfile.actualPosition - awayProfile.actualPosition
            : 0,
      }

      return {
        ...enrichedMatch,
        signalLabel: buildSignalLabel(enrichedMatch),
        notes: buildNotes(enrichedMatch),
        homePositionText: teamPositionText(homeProfile),
        awayPositionText: teamPositionText(awayProfile),
      }
    })
  )
}

export function buildUpcomingMatchInsights(matches) {
  if (!matches.length) {
    return {
      nextMatch: null,
      strongestLean: null,
      closestCall: null,
      biggestTableGap: null,
    }
  }

  return {
    nextMatch: matches[0],
    strongestLean: [...matches].sort((a, b) => b.modelConfidence - a.modelConfidence)[0],
    closestCall: [...matches].sort((a, b) => a.modelConfidence - b.modelConfidence)[0],
    biggestTableGap: [...matches].sort((a, b) => Math.abs(b.tablePositionGap) - Math.abs(a.tablePositionGap))[0],
  }
}
