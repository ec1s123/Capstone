// This code was generated with Codex.
export function derivePickCodeFromProbabilities(homeProb, drawProb, awayProb) {
  const probabilities = [
    ['H', homeProb],
    ['D', drawProb],
    ['A', awayProb],
  ]

  let topPick = probabilities[0]
  for (const probabilityEntry of probabilities.slice(1)) {
    if (probabilityEntry[1] > topPick[1]) {
      topPick = probabilityEntry
    }
  }
  return topPick[0]
}

export function deriveModelPickCode(match) {
  return derivePickCodeFromProbabilities(match.modelHomeProb, match.modelDrawProb, match.modelAwayProb)
}

export function deriveMarketPickCode(match) {
  return derivePickCodeFromProbabilities(match.marketHomeProb, match.marketDrawProb, match.marketAwayProb)
}

export function outcomeForClub(resultCode, isHome) {
  if (resultCode === 'D') return 'D'
  const isClubWin = (resultCode === 'H' && isHome) || (resultCode === 'A' && !isHome)
  return isClubWin ? 'W' : 'L'
}

export function projectExpectedRecord(row) {
  const played = row.Played
  let bestRecord = { won: 0, drawn: 0, lost: played, points: 0 }
  let bestKey = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, 0, 0]

  for (let won = 0; won <= played; won += 1) {
    for (let drawn = 0; drawn <= played - won; drawn += 1) {
      const lost = played - won - drawn
      const points = won * 3 + drawn
      const pointsGap = Math.abs(points - row.ExpectedPoints)
      const shapeGap =
        Math.abs(won - row.expectedWins) +
        Math.abs(drawn - row.expectedDraws) +
        Math.abs(lost - row.expectedLosses)
      const score = pointsGap * 4 + shapeGap
      const key = [score, pointsGap, shapeGap, -won, -drawn]

      let isBetter = false
      for (let index = 0; index < key.length; index += 1) {
        if (key[index] < bestKey[index]) {
          isBetter = true
          break
        }
        if (key[index] > bestKey[index]) break
      }

      if (isBetter) {
        bestKey = key
        bestRecord = { won, drawn, lost, points }
      }
    }
  }

  return bestRecord
}

export function emptyForm() {
  return [null, null, null, null, null]
}

export function normalizeRecentForm(results) {
  const form = results.slice(-5)
  while (form.length < 5) form.unshift(null)
  return form
}

export function sortFixturesChronologically(fixtures) {
  return [...fixtures].sort(
    (a, b) =>
      a.matchDate.localeCompare(b.matchDate) ||
      a.homeTeam.localeCompare(b.homeTeam) ||
      a.awayTeam.localeCompare(b.awayTeam)
  )
}

export function filterFixturesByGameweek(fixtures, gameweekLimit) {
  const safeLimit = Math.max(1, Math.min(38, Number(gameweekLimit) || 38))
  const orderedFixtures = sortFixturesChronologically(fixtures)
  const playedByTeam = new Map()
  const limitedFixtures = []

  orderedFixtures.forEach((fixture) => {
    const nextHomePlayed = (playedByTeam.get(fixture.homeTeam) ?? 0) + 1
    const nextAwayPlayed = (playedByTeam.get(fixture.awayTeam) ?? 0) + 1
    playedByTeam.set(fixture.homeTeam, nextHomePlayed)
    playedByTeam.set(fixture.awayTeam, nextAwayPlayed)

    if (nextHomePlayed <= safeLimit && nextAwayPlayed <= safeLimit) {
      limitedFixtures.push(fixture)
    }
  })

  return limitedFixtures
}

export function getSeasonCurrentGameweek(fixtures) {
  if (!fixtures.length) return 1
  const playedByTeam = new Map()
  sortFixturesChronologically(fixtures).forEach((fixture) => {
    playedByTeam.set(fixture.homeTeam, (playedByTeam.get(fixture.homeTeam) ?? 0) + 1)
    playedByTeam.set(fixture.awayTeam, (playedByTeam.get(fixture.awayTeam) ?? 0) + 1)
  })
  return Math.max(...playedByTeam.values())
}

function safeChartValue(value) {
  return Number.isFinite(value) ? value : 0
}

export function buildActualTable(fixtures, allTeams) {
  const tableByTeam = new Map()
  const orderedFixtures = sortFixturesChronologically(fixtures)

  const ensureTeam = (team) => {
    if (!tableByTeam.has(team)) {
      tableByTeam.set(team, {
        team,
        played: 0,
        won: 0,
        drawn: 0,
        lost: 0,
        points: 0,
        goalsFor: 0,
        goalsAgainst: 0,
        formResults: [],
      })
    }
    return tableByTeam.get(team)
  }

  allTeams.forEach((team) => {
    ensureTeam(team)
  })

  orderedFixtures.forEach((fixture) => {
    const homeRow = ensureTeam(fixture.homeTeam)
    const awayRow = ensureTeam(fixture.awayTeam)

    homeRow.played += 1
    awayRow.played += 1
    homeRow.goalsFor += safeChartValue(fixture.homeGoals)
    homeRow.goalsAgainst += safeChartValue(fixture.awayGoals)
    awayRow.goalsFor += safeChartValue(fixture.awayGoals)
    awayRow.goalsAgainst += safeChartValue(fixture.homeGoals)

    const homeOutcome = outcomeForClub(fixture.fullTimeResult, true)
    const awayOutcome = outcomeForClub(fixture.fullTimeResult, false)
    homeRow.formResults.push(homeOutcome)
    awayRow.formResults.push(awayOutcome)

    if (homeOutcome === 'W') {
      homeRow.won += 1
      homeRow.points += 3
    }
    if (homeOutcome === 'D') {
      homeRow.drawn += 1
      homeRow.points += 1
    }
    if (homeOutcome === 'L') homeRow.lost += 1

    if (awayOutcome === 'W') {
      awayRow.won += 1
      awayRow.points += 3
    }
    if (awayOutcome === 'D') {
      awayRow.drawn += 1
      awayRow.points += 1
    }
    if (awayOutcome === 'L') awayRow.lost += 1
  })

  return [...tableByTeam.values()]
    .map((row) => ({
      team: row.team,
      played: row.played,
      won: row.won,
      drawn: row.drawn,
      lost: row.lost,
      points: row.points,
      goalsFor: row.goalsFor,
      goalsAgainst: row.goalsAgainst,
      goalDifference: row.goalsFor - row.goalsAgainst,
      form: normalizeRecentForm(row.formResults),
    }))
    .sort(
      (a, b) =>
        b.points - a.points ||
        b.goalDifference - a.goalDifference ||
        b.goalsFor - a.goalsFor ||
        a.team.localeCompare(b.team)
    )
}

export function buildModelOutputTable(fixtures) {
  const tableByTeam = new Map()

  const ensureTeam = (team) => {
    if (!tableByTeam.has(team)) {
      tableByTeam.set(team, {
        Team: team,
        Played: 0,
        Won: 0,
        Drawn: 0,
        Lost: 0,
        Points: 0,
        ExpectedPoints: 0,
        expectedWins: 0,
        expectedDraws: 0,
        expectedLosses: 0,
        goalsFor: 0,
        goalsAgainst: 0,
        formResults: [],
      })
    }
    return tableByTeam.get(team)
  }

  const orderedFixtures = sortFixturesChronologically(fixtures)

  orderedFixtures.forEach((fixture) => {
    const homeRow = ensureTeam(fixture.homeTeam)
    const awayRow = ensureTeam(fixture.awayTeam)

    homeRow.Played += 1
    awayRow.Played += 1
    homeRow.goalsFor += safeChartValue(fixture.homeGoals)
    homeRow.goalsAgainst += safeChartValue(fixture.awayGoals)
    awayRow.goalsFor += safeChartValue(fixture.awayGoals)
    awayRow.goalsAgainst += safeChartValue(fixture.homeGoals)

    homeRow.expectedWins += fixture.modelHomeProb
    homeRow.expectedDraws += fixture.modelDrawProb
    homeRow.expectedLosses += fixture.modelAwayProb
    homeRow.ExpectedPoints += fixture.modelHomeProb * 3 + fixture.modelDrawProb

    awayRow.expectedWins += fixture.modelAwayProb
    awayRow.expectedDraws += fixture.modelDrawProb
    awayRow.expectedLosses += fixture.modelHomeProb
    awayRow.ExpectedPoints += fixture.modelAwayProb * 3 + fixture.modelDrawProb

    const modelPick = deriveModelPickCode(fixture)
    homeRow.formResults.push(outcomeForClub(modelPick, true))
    awayRow.formResults.push(outcomeForClub(modelPick, false))
  })

  return [...tableByTeam.values()]
    .map((row) => {
      const projected = projectExpectedRecord(row)
      const form = normalizeRecentForm(row.formResults)

      return {
        Team: row.Team,
        Played: row.Played,
        Won: projected.won,
        Drawn: projected.drawn,
        Lost: projected.lost,
        Points: projected.points,
        ExpectedPoints: Number(row.ExpectedPoints.toFixed(2)),
        ExpectedWins: Number(row.expectedWins.toFixed(2)),
        ExpectedDraws: Number(row.expectedDraws.toFixed(2)),
        ExpectedLosses: Number(row.expectedLosses.toFixed(2)),
        GoalsFor: row.goalsFor,
        GoalsAgainst: row.goalsAgainst,
        GoalDifference: row.goalsFor - row.goalsAgainst,
        Form: form,
      }
    })
    .sort((a, b) => b.ExpectedPoints - a.ExpectedPoints || b.Points - a.Points || a.Team.localeCompare(b.Team))
    .map((row, index) => ({ ...row, Position: index + 1 }))
}
