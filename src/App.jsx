import { useEffect, useMemo, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

import { AppNavigation } from './components/layout/AppNavigation'
import { defaultMatchColumnVisibility, MATCH_COLUMN_STORAGE_KEY, readCachedMatchColumnVisibility } from './constants/matchColumns'
import { teamList } from './data/placeholder'
import eplFinalRaw from './data/epl_final.csv?raw'
import premOddsPredictionsRaw from './data/prem_odds_predictions_all.csv?raw'
import { ClubPage } from './pages/ClubPage'
import { MatchesPage } from './pages/MatchesPage'
import { MethodologyPage } from './pages/MethodologyPage'
import { ModelOutputPage } from './pages/ModelOutputPage'
import { OverviewPage } from './pages/OverviewPage'
import { TablesPage } from './pages/TablesPage'
import { normalizeTeamName } from './lib/teamUtils'
import { parsePredictionFixtures, seasonStartFromLabel } from './lib/predictionData'
import {
  buildActualTable,
  buildModelOutputTable,
  deriveModelPickCode,
  emptyForm,
  filterFixturesByGameweek,
  getSeasonCurrentGameweek,
  outcomeForClub,
  sortFixturesChronologically,
} from './lib/standings'

const rawSeasonOddsCsvModules = import.meta.glob('../Prem-2026-2003/*.csv', {
  eager: true,
  import: 'default',
  query: '?raw',
})

export default function App() {
  const [selectedGameweek, setSelectedGameweek] = useState(38)
  const [favoriteTeam, setFavoriteTeam] = useState(teamList[0].team)
  const [selectedClub, setSelectedClub] = useState('')
  const [selectedSeason, setSelectedSeason] = useState('')
  const [matchColumnVisibility, setMatchColumnVisibility] = useState(() => readCachedMatchColumnVisibility())

  const predictionFixtures = useMemo(
    () => parsePredictionFixtures(premOddsPredictionsRaw, eplFinalRaw, rawSeasonOddsCsvModules),
    []
  )

  const seasonOptions = useMemo(() => {
    const uniqueSeasons = [...new Set(predictionFixtures.map((fixture) => fixture.season).filter(Boolean))]
    return uniqueSeasons.sort((a, b) => seasonStartFromLabel(b) - seasonStartFromLabel(a) || b.localeCompare(a))
  }, [predictionFixtures])

  useEffect(() => {
    if (!seasonOptions.length) return
    if (!selectedSeason || !seasonOptions.includes(selectedSeason)) {
      setSelectedSeason(seasonOptions[0])
    }
  }, [seasonOptions, selectedSeason])

  const activeSeason = seasonOptions.includes(selectedSeason) ? selectedSeason : (seasonOptions[0] ?? '')

  useEffect(() => {
    if (typeof window === 'undefined') return
    window.localStorage.setItem(MATCH_COLUMN_STORAGE_KEY, JSON.stringify(matchColumnVisibility))
  }, [matchColumnVisibility])

  const toggleMatchColumn = (columnKey) => {
    setMatchColumnVisibility((current) => ({
      ...current,
      [columnKey]: !(current[columnKey] !== false),
    }))
  }

  const resetMatchColumns = () => {
    setMatchColumnVisibility(defaultMatchColumnVisibility())
  }

  const seasonFixtures = useMemo(
    () => predictionFixtures.filter((fixture) => fixture.season === activeSeason),
    [predictionFixtures, activeSeason]
  )

  useEffect(() => {
    if (!activeSeason) return
    setSelectedGameweek(getSeasonCurrentGameweek(seasonFixtures))
  }, [activeSeason, seasonFixtures])

  const modelOutputTable = useMemo(() => buildModelOutputTable(seasonFixtures), [seasonFixtures])

  const seasonMatches = useMemo(
    () =>
      sortFixturesChronologically(seasonFixtures).reverse().map((match, index) => {
        const modelPickCode = deriveModelPickCode(match)
        const modelConfidence =
          modelPickCode === 'H'
            ? match.modelHomeProb
            : modelPickCode === 'D'
              ? match.modelDrawProb
              : match.modelAwayProb
        const goalMargin =
          Number.isFinite(match.homeGoals) && Number.isFinite(match.awayGoals)
            ? Math.abs(match.homeGoals - match.awayGoals)
            : null

        return {
          ...match,
          id: `${match.id}-${index}`,
          modelPickCode,
          modelConfidence,
          predictionCorrect: modelPickCode === match.fullTimeResult,
          goalMargin,
        }
      }),
    [seasonFixtures]
  )

  const matchInsights = useMemo(() => {
    if (!seasonMatches.length) {
      return {
        highestConfidence: null,
        lowestConfidence: null,
        highestConfidenceMiss: null,
        biggestGoalMargin: null,
      }
    }

    const highestConfidence = [...seasonMatches].sort((a, b) => b.modelConfidence - a.modelConfidence)[0]
    const lowestConfidence = [...seasonMatches].sort((a, b) => a.modelConfidence - b.modelConfidence)[0]
    const highestConfidenceMiss =
      [...seasonMatches]
        .filter((match) => !match.predictionCorrect)
        .sort((a, b) => b.modelConfidence - a.modelConfidence)[0] ?? null
    const biggestGoalMargin =
      [...seasonMatches]
        .filter((match) => Number.isFinite(match.goalMargin))
        .sort((a, b) => b.goalMargin - a.goalMargin)[0] ?? null

    return {
      highestConfidence,
      lowestConfidence,
      highestConfidenceMiss,
      biggestGoalMargin,
    }
  }, [seasonMatches])

  const availableClubs = useMemo(() => {
    const clubs = new Set()
    seasonFixtures.forEach((fixture) => {
      clubs.add(fixture.homeTeam)
      clubs.add(fixture.awayTeam)
    })
    return [...clubs].sort((a, b) => a.localeCompare(b))
  }, [seasonFixtures])

  const activeClub = availableClubs.includes(selectedClub) ? selectedClub : (availableClubs[0] ?? '')

  const clubFixtures = useMemo(() => {
    if (!activeClub) return []

    return seasonFixtures
      .filter((match) => match.homeTeam === activeClub || match.awayTeam === activeClub)
      .map((match, index) => {
        const isHome = match.homeTeam === activeClub
        const modelPickCode = deriveModelPickCode(match)
        const modelConfidence =
          modelPickCode === 'H'
            ? match.modelHomeProb
            : modelPickCode === 'D'
              ? match.modelDrawProb
              : match.modelAwayProb
        const actualOutcome = outcomeForClub(match.fullTimeResult, isHome)
        const modelOutcome = outcomeForClub(modelPickCode, isHome)

        return {
          ...match,
          id: `${match.id}-${index}`,
          matchDate: match.matchDate,
          opponent: isHome ? match.awayTeam : match.homeTeam,
          venue: isHome ? 'Home' : 'Away',
          modelPickCode,
          actualOutcome,
          modelOutcome,
          modelConfidence,
          winProbability: isHome ? match.modelHomeProb : match.modelAwayProb,
          drawProbability: match.modelDrawProb,
          lossProbability: isHome ? match.modelAwayProb : match.modelHomeProb,
          predictionCorrect: modelPickCode === match.fullTimeResult,
        }
      })
      .sort((a, b) => a.matchDate.localeCompare(b.matchDate))
      .reverse()
  }, [activeClub, seasonFixtures])

  const clubSummary = useMemo(() => {
    if (!clubFixtures.length) return null

    const wins = clubFixtures.filter((fixture) => fixture.actualOutcome === 'W').length
    const draws = clubFixtures.filter((fixture) => fixture.actualOutcome === 'D').length
    const losses = clubFixtures.filter((fixture) => fixture.actualOutcome === 'L').length
    const modelHits = clubFixtures.filter((fixture) => fixture.predictionCorrect).length
    const confidenceTotal = clubFixtures.reduce((sum, fixture) => sum + fixture.modelConfidence, 0)
    const expectedWins = clubFixtures.reduce((sum, fixture) => sum + fixture.winProbability, 0)
    const expectedDraws = clubFixtures.reduce((sum, fixture) => sum + fixture.drawProbability, 0)
    const expectedLosses = clubFixtures.reduce((sum, fixture) => sum + fixture.lossProbability, 0)
    const actualPoints = wins * 3 + draws
    const expectedPoints = expectedWins * 3 + expectedDraws

    return {
      played: clubFixtures.length,
      wins,
      draws,
      losses,
      actualPoints,
      expectedWins,
      expectedDraws,
      expectedLosses,
      expectedPoints,
      winDelta: wins - expectedWins,
      drawDelta: draws - expectedDraws,
      lossDelta: losses - expectedLosses,
      pointDelta: actualPoints - expectedPoints,
      modelAccuracy: modelHits / clubFixtures.length,
      averageConfidence: confidenceTotal / clubFixtures.length,
    }
  }, [clubFixtures])

  const seasonTeams = useMemo(() => [...availableClubs], [availableClubs])

  useEffect(() => {
    if (!seasonTeams.length) return
    const normalizedFavoriteTeam = normalizeTeamName(favoriteTeam)
    if (!seasonTeams.includes(normalizedFavoriteTeam)) {
      setFavoriteTeam(seasonTeams[0])
    }
  }, [seasonTeams, favoriteTeam])

  const fixturesThroughGameweek = useMemo(
    () => filterFixturesByGameweek(seasonFixtures, selectedGameweek),
    [seasonFixtures, selectedGameweek]
  )

  const { currentTable, predictedTable, topOver, topUnder, favoriteSnapshot } = useMemo(() => {
    const actualRows = buildActualTable(fixturesThroughGameweek, seasonTeams)
    const predictedRows = buildModelOutputTable(fixturesThroughGameweek)
    const actualByTeam = new Map(actualRows.map((row) => [normalizeTeamName(row.team), row]))
    const predictedByTeam = new Map(
      predictedRows.map((row) => [normalizeTeamName(row.Team), row])
    )

    const mergedRows = seasonTeams.map((team) => {
      const actualRow = actualByTeam.get(team) ?? {
        team,
        played: 0,
        points: 0,
        form: emptyForm(),
      }
      const predictedRow = predictedByTeam.get(team)
      return {
        team,
        played: actualRow.played,
        points: actualRow.points,
        predictedPoints: predictedRow?.Points ?? 0,
        form: actualRow.form,
        predictedForm: predictedRow?.Form ?? emptyForm(),
      }
    })

    const currentTable = [...mergedRows]
      .sort((a, b) => b.points - a.points || b.predictedPoints - a.predictedPoints || a.team.localeCompare(b.team))
      .map((row, index) => ({ ...row, position: index + 1 }))

    const predictedTable = [...mergedRows]
      .sort((a, b) => b.predictedPoints - a.predictedPoints || b.points - a.points || a.team.localeCompare(b.team))
      .map((row, index) => ({
        ...row,
        position: index + 1,
        delta: row.predictedPoints - row.points,
      }))

    const deltas = mergedRows
      .map((row) => ({
        team: row.team,
        points: row.points,
        predictedPoints: row.predictedPoints,
        delta: row.points - row.predictedPoints,
      }))
      .sort((a, b) => b.delta - a.delta)

    const normalizedFavoriteTeam = normalizeTeamName(favoriteTeam)
    const favoriteCurrent = mergedRows.find((row) => normalizeTeamName(row.team) === normalizedFavoriteTeam)
    const favoritePredicted = mergedRows.find((row) => normalizeTeamName(row.team) === normalizedFavoriteTeam)
    const favoriteSnapshot =
      favoriteCurrent && favoritePredicted
        ? {
            points: favoriteCurrent.points,
            predictedPoints: favoritePredicted.predictedPoints,
            delta: favoriteCurrent.points - favoritePredicted.predictedPoints,
          }
        : null

    return {
      currentTable,
      predictedTable,
      topOver: deltas.slice(0, 3),
      topUnder: [...deltas].reverse().slice(0, 3),
      favoriteSnapshot,
    }
  }, [favoriteTeam, fixturesThroughGameweek, seasonTeams])

  return (
    <div className="relative min-h-screen overflow-x-hidden">
      <AppNavigation />
      <main className="relative z-10 mx-auto w-full max-w-[1440px] px-3 py-8 sm:px-4 md:py-10 lg:px-6 lg:py-12">
        <Routes>
          <Route
            path="/"
            element={
              <OverviewPage
                season={activeSeason}
                seasonOptions={seasonOptions}
                onSeasonChange={setSelectedSeason}
                topOver={topOver}
                topUnder={topUnder}
                favoriteTeam={favoriteTeam}
                favoriteSnapshot={favoriteSnapshot}
                onFavoriteTeamChange={setFavoriteTeam}
                teamOptions={seasonTeams}
              />
            }
          />
          <Route
            path="/tables"
            element={
              <TablesPage
                season={activeSeason}
                seasonOptions={seasonOptions}
                onSeasonChange={setSelectedSeason}
                currentTable={currentTable}
                predictedTable={predictedTable}
                favoriteTeam={favoriteTeam}
                gameweek={selectedGameweek}
                onGameweekChange={setSelectedGameweek}
              />
            }
          />
          <Route
            path="/matches"
            element={
              <MatchesPage
                season={activeSeason}
                seasonOptions={seasonOptions}
                onSeasonChange={setSelectedSeason}
                matches={seasonMatches}
                insights={matchInsights}
                columnVisibility={matchColumnVisibility}
                onToggleColumn={toggleMatchColumn}
                onResetColumns={resetMatchColumns}
              />
            }
          />
          <Route
            path="/club"
            element={
              <ClubPage
                season={activeSeason}
                seasonOptions={seasonOptions}
                onSeasonChange={setSelectedSeason}
                clubs={availableClubs}
                selectedClub={activeClub}
                onSelectedClubChange={setSelectedClub}
                clubFixtures={clubFixtures}
                clubSummary={clubSummary}
              />
            }
          />
          <Route path="/methodology" element={<MethodologyPage />} />
          <Route
            path="/model-output"
            element={
              <ModelOutputPage
                favoriteTeam={favoriteTeam}
                matches={seasonMatches}
                modelOutputTable={modelOutputTable}
                season={activeSeason}
                seasonOptions={seasonOptions}
                onSeasonChange={setSelectedSeason}
              />
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  )
}
