import { useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  Activity,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  Columns3,
  Home,
  Sparkles,
  TrendingDown,
  TrendingUp,
  X,
} from 'lucide-react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'

import { Badge } from './components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from './components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from './components/ui/table'
import { teamList } from './data/placeholder'
import eplFinalRaw from './data/epl_final.csv?raw'
import premOddsPredictionsRaw from './data/prem_odds_predictions_all.csv?raw'
import { cn } from './lib/utils'

const gameweeks = Array.from({ length: 38 }, (_, index) => index + 1)

const navItems = [
  { to: '/', label: 'Overview' },
  { to: '/tables', label: 'Tables' },
  { to: '/matches', label: 'Matches' },
  { to: '/club', label: 'Club' },
  { to: '/methodology', label: 'Methodology' },
  { to: '/model-output', label: 'Model Output' },
]

const MATCH_COLUMN_STORAGE_KEY = 'premier_predict.matches.columns.v1'
const MATCH_COLUMN_DEFINITIONS = [
  { key: 'date', label: 'Date' },
  { key: 'home', label: 'Home' },
  { key: 'away', label: 'Away' },
  { key: 'score', label: 'Score' },
  { key: 'result', label: 'Result' },
  { key: 'modelPick', label: 'Model Pick' },
  { key: 'confidence', label: 'Confidence' },
  { key: 'prediction', label: 'Prediction' },
]

const rawSeasonOddsCsvModules = import.meta.glob('../Prem-2026-2003/*.csv', {
  eager: true,
  import: 'default',
  query: '?raw',
})

const methodology = [
  {
    title: 'Historical Performance',
    description: 'Long-term team patterns, matchup history, and season-level trends shape the baseline view.',
    icon: Activity,
  },
  {
    title: 'Recent Form Tracking',
    description: 'Short-term momentum, consistency, and changes in output are factored into every fixture forecast.',
    icon: TrendingUp,
  },
  {
    title: 'Match Context',
    description: 'Home and away strength, scoring profiles, and defensive stability refine the prediction for each game.',
    icon: Home,
  },
  {
    title: 'Model-Driven Probabilities',
    description: 'Structured model outputs turn raw football data into clearer, more actionable prediction signals.',
    icon: BrainCircuit,
  },
]

const teamAliases = {
  'Man City': 'Manchester City',
  'Man United': 'Manchester United',
  Newcastle: 'Newcastle United',
  "Nott'm Forest": 'Nottingham Forest',
  QPR: 'Queens Park Rangers',
  'West Brom': 'West Bromwich Albion',
  Wolves: 'Wolverhampton Wanderers',
}

const teamDomains = {
  Arsenal: 'arsenal.com',
  'Aston Villa': 'avfc.co.uk',
  Bournemouth: 'afcb.co.uk',
  Brentford: 'brentfordfc.com',
  Brighton: 'brightonandhovealbion.com',
  Burnley: 'burnleyfootballclub.com',
  Chelsea: 'chelseafc.com',
  'Crystal Palace': 'cpfc.co.uk',
  Everton: 'evertonfc.com',
  Fulham: 'fulhamfc.com',
  Ipswich: 'itfc.co.uk',
  Leeds: 'leedsunited.com',
  Leicester: 'lcfc.com',
  Liverpool: 'liverpoolfc.com',
  'Luton Town': 'lutontown.co.uk',
  'Manchester City': 'mancity.com',
  'Manchester United': 'manutd.com',
  'Newcastle United': 'nufc.co.uk',
  'Nottingham Forest': 'nottinghamforest.co.uk',
  Southampton: 'southamptonfc.com',
  'Sheffield United': 'sufc.co.uk',
  Sunderland: 'safc.com',
  Tottenham: 'tottenhamhotspur.com',
  'West Ham': 'whufc.com',
  Wolves: 'wolves.co.uk',
  'Wolverhampton Wanderers': 'wolves.co.uk',
}

const toneMap = {
  current: {
    source: 'Live Data',
    card: 'border-sky-200 bg-white',
    badge: 'border-sky-200 bg-sky-50 text-sky-700',
    header: 'bg-sky-50/80',
    favorite: 'bg-sky-50 hover:bg-sky-100/70',
    pos: 'text-sky-700',
    accentBorder: 'border-sky-200',
  },
  predicted: {
    source: 'Model Forecast',
    card: 'border-emerald-200 bg-white',
    badge: 'border-emerald-200 bg-emerald-50 text-emerald-700',
    header: 'bg-emerald-50/80',
    favorite: 'bg-emerald-50 hover:bg-emerald-100/70',
    pos: 'text-emerald-700',
    accentBorder: 'border-emerald-200',
  },
}

function normalizeTeamName(team) {
  return teamAliases[team] ?? team
}

function getClubLogoUrl(team) {
  const normalizedTeam = normalizeTeamName(team)
  const domain = teamDomains[normalizedTeam]
  if (!domain) return null
  return `https://www.google.com/s2/favicons?domain=${domain}&sz=128`
}

function getTeamInitials(team) {
  return team
    .split(' ')
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase())
    .slice(0, 2)
    .join('')
}

function deltaClass(delta) {
  if (delta > 0) return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (delta < 0) return 'border-rose-200 bg-rose-50 text-rose-700'
  return 'border-slate-200 bg-slate-100 text-slate-600'
}

function formResultClass(result) {
  if (result === 'W') return 'bg-emerald-700 text-white'
  if (result === 'L') return 'bg-rose-600 text-white'
  if (result === 'D') return 'bg-slate-500 text-white'
  return 'bg-slate-200 text-slate-400'
}

const outcomeLabelMap = {
  W: 'Win',
  D: 'Draw',
  L: 'Loss',
}

const matchOutcomeLabelMap = {
  H: 'Home Win',
  D: 'Draw',
  A: 'Away Win',
}

function defaultMatchColumnVisibility() {
  return Object.fromEntries(MATCH_COLUMN_DEFINITIONS.map((column) => [column.key, true]))
}

function sanitizeMatchColumnVisibility(rawVisibility) {
  const defaults = defaultMatchColumnVisibility()
  if (!rawVisibility || typeof rawVisibility !== 'object') return defaults

  return MATCH_COLUMN_DEFINITIONS.reduce((accumulator, column) => {
    const rawValue = rawVisibility[column.key]
    accumulator[column.key] = typeof rawValue === 'boolean' ? rawValue : defaults[column.key]
    return accumulator
  }, {})
}

function readCachedMatchColumnVisibility() {
  if (typeof window === 'undefined') return defaultMatchColumnVisibility()
  try {
    const cachedValue = window.localStorage.getItem(MATCH_COLUMN_STORAGE_KEY)
    if (!cachedValue) return defaultMatchColumnVisibility()
    return sanitizeMatchColumnVisibility(JSON.parse(cachedValue))
  } catch {
    return defaultMatchColumnVisibility()
  }
}

function seasonStartFromLabel(seasonLabel) {
  const [startToken] = String(seasonLabel).split('/')
  const parsed = Number.parseInt(startToken, 10)
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY
}

function inferSeasonFromMatchDate(dateValue) {
  const parsedDate = new Date(dateValue)
  if (Number.isNaN(parsedDate.getTime())) return ''
  const year = parsedDate.getUTCFullYear()
  const month = parsedDate.getUTCMonth() + 1
  const seasonStart = month >= 7 ? year : year - 1
  return `${seasonStart}/${String(seasonStart + 1).slice(-2)}`
}

function parseNumber(value, fallback = 0) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function parseNullableNumber(value) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function fixtureLookupKey(matchDate, homeTeam, awayTeam) {
  return `${matchDate}::${normalizeTeamName(homeTeam)}::${normalizeTeamName(awayTeam)}`
}

function normalizeFixtureDate(dateValue) {
  const rawValue = String(dateValue ?? '').trim()
  if (!rawValue) return ''
  if (/^\d{4}-\d{2}-\d{2}$/.test(rawValue)) return rawValue

  const slashParts = rawValue.split('/')
  if (slashParts.length === 3) {
    const [dayToken, monthToken, yearToken] = slashParts
    const day = Number.parseInt(dayToken, 10)
    const month = Number.parseInt(monthToken, 10)
    const rawYear = Number.parseInt(yearToken, 10)
    if (Number.isFinite(day) && Number.isFinite(month) && Number.isFinite(rawYear)) {
      const year = rawYear < 100 ? 2000 + rawYear : rawYear
      return `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`
    }
  }

  const parsedDate = new Date(rawValue)
  if (Number.isNaN(parsedDate.getTime())) return rawValue
  return parsedDate.toISOString().slice(0, 10)
}

const bookmakerLabelMap = {
  '1XB': '1xBet',
  B365: 'Bet365',
  BF: 'Betfair',
  BFD: 'Betfair Sportsbook',
  BFE: 'Betfair Exchange',
  BMGM: 'BetMGM',
  BV: 'BetVictor',
  BW: 'Bwin',
  CL: 'Coral',
  GB: 'Gamebookers',
  IW: 'Interwetten',
  LB: 'Ladbrokes',
  PS: 'Pinnacle',
  SB: 'Sportingbet',
  SO: 'Sporting Odds',
  VC: 'VC Bet',
  WH: 'William Hill',
  Max: 'Market Max',
  Avg: 'Market Average',
  BbMx: 'Market Max',
  BbAv: 'Market Average',
}

function bookmakerLabel(code) {
  return bookmakerLabelMap[code] ?? code
}

function parseSupplementalMatchStats(rawCsv) {
  const rows = rawCsv.trim().split(/\r?\n/)
  if (rows.length < 2) return new Map()

  const headers = rows[0].split(',').map((header) => header.replace(/^\uFEFF/, '').trim())
  const headerIndex = Object.fromEntries(headers.map((header, index) => [header, index]))
  const readValue = (values, key) => values[headerIndex[key]] ?? ''
  const statsByFixture = new Map()

  rows
    .slice(1)
    .filter(Boolean)
    .forEach((line) => {
      const values = line.split(',')
      const matchDate = readValue(values, 'MatchDate')
      const homeTeam = normalizeTeamName(readValue(values, 'HomeTeam'))
      const awayTeam = normalizeTeamName(readValue(values, 'AwayTeam'))

      statsByFixture.set(fixtureLookupKey(matchDate, homeTeam, awayTeam), {
        halfTimeHomeGoals: parseNullableNumber(readValue(values, 'HalfTimeHomeGoals')),
        halfTimeAwayGoals: parseNullableNumber(readValue(values, 'HalfTimeAwayGoals')),
        halfTimeResult: readValue(values, 'HalfTimeResult'),
        homeFouls: parseNullableNumber(readValue(values, 'HomeFouls')),
        awayFouls: parseNullableNumber(readValue(values, 'AwayFouls')),
      })
    })

  return statsByFixture
}

function collectBookmakerOdds(headers) {
  const bookmakerColumns = new Map()

  headers.forEach((header) => {
    const match = header.match(/^(.+?)(C?)(H|D|A)$/)
    if (!match) return

    const [, rawCode, closeMarker, outcome] = match
    if (rawCode.includes('AH') || rawCode.includes('>') || rawCode.includes('<')) return

    const marketType = closeMarker === 'C' ? 'closing' : 'opening'
    const current = bookmakerColumns.get(rawCode) ?? {
      code: rawCode,
      opening: {},
      closing: {},
    }
    current[marketType][outcome] = header
    bookmakerColumns.set(rawCode, current)
  })

  return [...bookmakerColumns.values()].filter(
    (bookmaker) =>
      (bookmaker.opening.H && bookmaker.opening.D && bookmaker.opening.A) ||
      (bookmaker.closing.H && bookmaker.closing.D && bookmaker.closing.A)
  )
}

function readBookmakerOdds(values, readValue, bookmakerColumns) {
  return bookmakerColumns
    .map((bookmaker) => ({
      code: bookmaker.code,
      label: bookmakerLabel(bookmaker.code),
      home: parseNullableNumber(readValue(values, bookmaker.opening.H)),
      draw: parseNullableNumber(readValue(values, bookmaker.opening.D)),
      away: parseNullableNumber(readValue(values, bookmaker.opening.A)),
      closingHome: parseNullableNumber(readValue(values, bookmaker.closing.H)),
      closingDraw: parseNullableNumber(readValue(values, bookmaker.closing.D)),
      closingAway: parseNullableNumber(readValue(values, bookmaker.closing.A)),
    }))
    .filter(
      (bookmaker) =>
        [bookmaker.home, bookmaker.draw, bookmaker.away, bookmaker.closingHome, bookmaker.closingDraw, bookmaker.closingAway]
          .some(Number.isFinite)
    )
    .sort((a, b) => {
      const aAggregate = a.label.startsWith('Market ')
      const bAggregate = b.label.startsWith('Market ')
      if (aAggregate !== bAggregate) return aAggregate ? 1 : -1
      return a.label.localeCompare(b.label)
    })
}

function parseRawSeasonMatchDetails(rawCsvModules) {
  const detailsByFixture = new Map()

  Object.values(rawCsvModules).forEach((rawCsv) => {
    const rows = String(rawCsv).trim().split(/\r?\n/)
    if (rows.length < 2) return

    const headers = rows[0].split(',').map((header) => header.replace(/^\uFEFF/, '').trim())
    const headerIndex = Object.fromEntries(headers.map((header, index) => [header, index]))
    const readValue = (values, key) => (key ? values[headerIndex[key]] ?? '' : '')
    const bookmakerColumns = collectBookmakerOdds(headers)

    rows
      .slice(1)
      .filter(Boolean)
      .forEach((line) => {
        const values = line.split(',')
        const matchDate = normalizeFixtureDate(readValue(values, 'Date'))
        const homeTeam = normalizeTeamName(readValue(values, 'HomeTeam'))
        const awayTeam = normalizeTeamName(readValue(values, 'AwayTeam'))
        if (!matchDate || !homeTeam || !awayTeam) return

        detailsByFixture.set(fixtureLookupKey(matchDate, homeTeam, awayTeam), {
          halfTimeHomeGoals: parseNullableNumber(readValue(values, 'HTHG')),
          halfTimeAwayGoals: parseNullableNumber(readValue(values, 'HTAG')),
          halfTimeResult: readValue(values, 'HTR'),
          homeFouls: parseNullableNumber(readValue(values, 'HF')),
          awayFouls: parseNullableNumber(readValue(values, 'AF')),
          bookmakerOdds: readBookmakerOdds(values, readValue, bookmakerColumns),
        })
      })
  })

  return detailsByFixture
}

function fallbackBookmakerOddsFromPrediction(match) {
  return [
    {
      code: 'B365',
      label: bookmakerLabel('B365'),
      home: match.b365HomeOdds,
      draw: match.b365DrawOdds,
      away: match.b365AwayOdds,
      closingHome: null,
      closingDraw: null,
      closingAway: null,
    },
  ].filter((bookmaker) => [bookmaker.home, bookmaker.draw, bookmaker.away].some(Number.isFinite))
}

function parsePredictionFixtures(rawCsv, supplementalCsv = '', rawOddsCsvModules = {}) {
  const rows = rawCsv.trim().split(/\r?\n/)
  if (rows.length < 2) return []

  const headers = rows[0].split(',').map((header) => header.replace(/^\uFEFF/, '').trim())
  const headerIndex = Object.fromEntries(headers.map((header, index) => [header, index]))
  const readValue = (values, key) => values[headerIndex[key]] ?? ''
  const supplementalStats = supplementalCsv ? parseSupplementalMatchStats(supplementalCsv) : new Map()
  const rawSeasonDetails = parseRawSeasonMatchDetails(rawOddsCsvModules)

  return rows
    .slice(1)
    .filter(Boolean)
    .map((line) => {
      const values = line.split(',')
      const matchDate = readValue(values, 'MatchDate')
      const homeTeam = normalizeTeamName(readValue(values, 'HomeTeam'))
      const awayTeam = normalizeTeamName(readValue(values, 'AwayTeam'))
      const season = readValue(values, 'Season') || inferSeasonFromMatchDate(matchDate)
      const lookupKey = fixtureLookupKey(matchDate, homeTeam, awayTeam)
      const supplemental = {
        ...(supplementalStats.get(lookupKey) ?? {}),
        ...(rawSeasonDetails.get(lookupKey) ?? {}),
      }
      const baseMatch = {
        b365HomeOdds: parseNullableNumber(readValue(values, 'B365H')),
        b365DrawOdds: parseNullableNumber(readValue(values, 'B365D')),
        b365AwayOdds: parseNullableNumber(readValue(values, 'B365A')),
      }
      return {
        id: `${matchDate}-${homeTeam}-${awayTeam}`,
        season,
        matchDate,
        homeTeam,
        awayTeam,
        fullTimeResult: readValue(values, 'FTR'),
        homeGoals: parseNullableNumber(readValue(values, 'FTHG')),
        awayGoals: parseNullableNumber(readValue(values, 'FTAG')),
        homeShots: parseNullableNumber(readValue(values, 'HS')),
        awayShots: parseNullableNumber(readValue(values, 'AS')),
        homeShotsOnTarget: parseNullableNumber(readValue(values, 'HST')),
        awayShotsOnTarget: parseNullableNumber(readValue(values, 'AST')),
        homeCorners: parseNullableNumber(readValue(values, 'HC')),
        awayCorners: parseNullableNumber(readValue(values, 'AC')),
        homeYellowCards: parseNullableNumber(readValue(values, 'HY')),
        awayYellowCards: parseNullableNumber(readValue(values, 'AY')),
        homeRedCards: parseNullableNumber(readValue(values, 'HR')),
        awayRedCards: parseNullableNumber(readValue(values, 'AR')),
        ...baseMatch,
        marketHomeProb: parseNumber(readValue(values, 'MarketHomeProb')),
        marketDrawProb: parseNumber(readValue(values, 'MarketDrawProb')),
        marketAwayProb: parseNumber(readValue(values, 'MarketAwayProb')),
        modelHomeProb: parseNumber(readValue(values, 'ModelHomeProb')),
        modelDrawProb: parseNumber(readValue(values, 'ModelDrawProb')),
        modelAwayProb: parseNumber(readValue(values, 'ModelAwayProb')),
        ...supplemental,
        bookmakerOdds: supplemental.bookmakerOdds?.length
          ? supplemental.bookmakerOdds
          : fallbackBookmakerOddsFromPrediction(baseMatch),
      }
    })
}

function derivePickCodeFromProbabilities(homeProb, drawProb, awayProb) {
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

function deriveModelPickCode(match) {
  return derivePickCodeFromProbabilities(match.modelHomeProb, match.modelDrawProb, match.modelAwayProb)
}

function deriveMarketPickCode(match) {
  return derivePickCodeFromProbabilities(match.marketHomeProb, match.marketDrawProb, match.marketAwayProb)
}

function outcomeForClub(resultCode, isHome) {
  if (resultCode === 'D') return 'D'
  const isClubWin = (resultCode === 'H' && isHome) || (resultCode === 'A' && !isHome)
  return isClubWin ? 'W' : 'L'
}

function outcomeBadgeClass(outcome) {
  if (outcome === 'W') return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (outcome === 'L') return 'border-rose-200 bg-rose-50 text-rose-700'
  return 'border-slate-200 bg-slate-100 text-slate-700'
}

function confidenceBadgeClass(value) {
  if (value >= 0.6) return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (value >= 0.45) return 'border-amber-200 bg-amber-50 text-amber-700'
  return 'border-slate-200 bg-slate-100 text-slate-700'
}

function matchOutcomeBadgeClass(resultCode) {
  if (resultCode === 'H') return 'border-emerald-200 bg-emerald-50 text-emerald-700'
  if (resultCode === 'A') return 'border-rose-200 bg-rose-50 text-rose-700'
  return 'border-slate-200 bg-slate-100 text-slate-700'
}

function formatScoreline(homeGoals, awayGoals) {
  if (!Number.isFinite(homeGoals) || !Number.isFinite(awayGoals)) return '-'
  return `${homeGoals}-${awayGoals}`
}

function formatStatPair(homeValue, awayValue) {
  if (!Number.isFinite(homeValue) || !Number.isFinite(awayValue)) return '-'
  return `${homeValue}-${awayValue}`
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function formatOptionalPercent(value) {
  if (!Number.isFinite(value)) return '-'
  return formatPercent(value)
}

function formatOptionalStat(value, decimals = 0) {
  if (!Number.isFinite(value)) return '-'
  return value.toFixed(decimals)
}

function formatOdds(value) {
  if (!Number.isFinite(value)) return '-'
  return value.toFixed(2)
}

function formatSigned(value, decimals = 1) {
  const fixed = value.toFixed(decimals)
  return value > 0 ? `+${fixed}` : fixed
}

function projectExpectedRecord(row) {
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

function emptyForm() {
  return [null, null, null, null, null]
}

function normalizeRecentForm(results) {
  const form = results.slice(-5)
  while (form.length < 5) form.unshift(null)
  return form
}

function sortFixturesChronologically(fixtures) {
  return [...fixtures].sort(
    (a, b) =>
      a.matchDate.localeCompare(b.matchDate) ||
      a.homeTeam.localeCompare(b.homeTeam) ||
      a.awayTeam.localeCompare(b.awayTeam)
  )
}

function filterFixturesByGameweek(fixtures, gameweekLimit) {
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

function getSeasonCurrentGameweek(fixtures) {
  if (!fixtures.length) return 1
  const playedByTeam = new Map()
  sortFixturesChronologically(fixtures).forEach((fixture) => {
    playedByTeam.set(fixture.homeTeam, (playedByTeam.get(fixture.homeTeam) ?? 0) + 1)
    playedByTeam.set(fixture.awayTeam, (playedByTeam.get(fixture.awayTeam) ?? 0) + 1)
  })
  return Math.max(...playedByTeam.values())
}

function buildActualTable(fixtures, allTeams) {
  const tableByTeam = new Map()
  const orderedFixtures = sortFixturesChronologically(fixtures)

  const ensureTeam = (team) => {
    if (!tableByTeam.has(team)) {
      tableByTeam.set(team, {
        team,
        played: 0,
        points: 0,
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

    const homeOutcome = outcomeForClub(fixture.fullTimeResult, true)
    const awayOutcome = outcomeForClub(fixture.fullTimeResult, false)
    homeRow.formResults.push(homeOutcome)
    awayRow.formResults.push(awayOutcome)

    if (homeOutcome === 'W') homeRow.points += 3
    if (homeOutcome === 'D') homeRow.points += 1
    if (awayOutcome === 'W') awayRow.points += 3
    if (awayOutcome === 'D') awayRow.points += 1
  })

  return [...tableByTeam.values()]
    .map((row) => ({
      team: row.team,
      played: row.played,
      points: row.points,
      form: normalizeRecentForm(row.formResults),
    }))
    .sort((a, b) => b.points - a.points || a.team.localeCompare(b.team))
}

function buildModelOutputTable(fixtures) {
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
        Form: form,
      }
    })
    .sort((a, b) => b.ExpectedPoints - a.ExpectedPoints || b.Points - a.Points || a.Team.localeCompare(b.Team))
    .map((row, index) => ({ ...row, Position: index + 1 }))
}

function comparisonDeltaClass(delta, inverse = false) {
  const adjusted = inverse ? -delta : delta
  if (adjusted > 0.05) return 'text-emerald-700'
  if (adjusted < -0.05) return 'text-rose-700'
  return 'text-slate-600'
}

const logoSizeMap = {
  sm: {
    fallback: 'h-5 w-5 text-[8px]',
    image: 'h-5 w-5',
  },
  md: {
    fallback: 'h-6 w-6 text-[9px]',
    image: 'h-6 w-6',
  },
  lg: {
    fallback: 'h-10 w-10 text-xs',
    image: 'h-10 w-10',
  },
  xl: {
    fallback: 'h-14 w-14 text-sm',
    image: 'h-14 w-14',
  },
}

function ClubLogo({ team, size = 'md' }) {
  const [hasError, setHasError] = useState(false)
  const normalizedTeam = normalizeTeamName(team)
  const logoUrl = getClubLogoUrl(normalizedTeam)
  const initials = getTeamInitials(normalizedTeam)
  const sizeClass = logoSizeMap[size] ?? logoSizeMap.md

  if (!logoUrl || hasError) {
    return (
      <span
        className={cn(
          'inline-flex shrink-0 items-center justify-center rounded-full border border-slate-300 bg-white font-semibold text-slate-700',
          sizeClass.fallback
        )}
      >
        {initials}
      </span>
    )
  }

  return (
    <img
      src={logoUrl}
      alt={`${normalizedTeam} crest`}
      className={cn(
        'shrink-0 rounded-full border border-slate-200 bg-white object-cover p-[1px]',
        sizeClass.image
      )}
      loading="lazy"
      onError={() => setHasError(true)}
    />
  )
}

function FormChips({ results }) {
  const safeResults = Array.isArray(results) ? results : [null, null, null, null, null]

  return (
    <div className="flex items-center gap-1 whitespace-nowrap">
      {safeResults.map((result, index) => (
        <span
          key={`${result ?? 'na'}-${index}`}
          className={cn(
            'inline-flex h-5 w-5 items-center justify-center rounded-sm text-[9px] font-bold tracking-[0.04em]',
            formResultClass(result)
          )}
          aria-label={result ? `Result ${result}` : 'No result'}
          title={result ?? 'No result'}
        >
          {result ?? '-'}
        </span>
      ))}
    </div>
  )
}

function SeasonSelector({ season, seasonOptions, onSeasonChange, className }) {
  return (
    <div className={cn('space-y-2', className)}>
      <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Season</p>
      <Select value={season} onValueChange={onSeasonChange} disabled={!seasonOptions.length}>
        <SelectTrigger className="h-10 w-[160px] border-slate-300 bg-white text-sm font-semibold text-slate-900">
          <SelectValue placeholder="Select season" />
        </SelectTrigger>
        <SelectContent>
          {seasonOptions.map((seasonOption) => (
            <SelectItem key={`season-${seasonOption}`} value={seasonOption}>
              {seasonOption}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

function TableCard({ title, rows, favoriteTeam, mode, gameweek, onGameweekChange }) {
  const tone = toneMap[mode]
  const isCurrentMode = mode === 'current'
  const showDeltaColumn = mode === 'predicted'
  const teamColumnClass = showDeltaColumn ? 'w-[180px]' : 'w-[220px]'
  const formColumnClass = showDeltaColumn ? 'w-[132px]' : 'w-[148px]'

  return (
    <Card className={cn('h-full overflow-hidden border shadow-sm', tone.card)}>
      <CardHeader className="px-4 pb-4 pt-4 sm:px-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="mb-2 text-[11px] uppercase tracking-[0.2em] text-muted-foreground">{tone.eyebrow}</p>
            <CardTitle className="text-xl">{title}</CardTitle>
            <CardDescription className="mt-1 text-xs">
              Form (W = win, D = draw, L = loss) is shown as the last five matches, oldest to newest.
            </CardDescription>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Select value={String(gameweek)} onValueChange={(value) => onGameweekChange(Number(value))}>
              <SelectTrigger
                className={cn('h-9 w-[96px] bg-white text-xs text-slate-900', tone.accentBorder)}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {gameweeks.map((week) => (
                  <SelectItem key={`${mode}-gw-${week}`} value={String(week)}>
                    GW {week}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Badge variant="outline" className={cn('uppercase tracking-[0.14em]', tone.badge)}>
              {tone.source}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-4 pb-4 pt-0 sm:px-5">
        <div className="overflow-x-auto overflow-y-hidden rounded-lg border border-slate-200">
          <Table className="table-fixed min-w-full">
            <TableHeader className={tone.header}>
              <TableRow className="border-b-0 hover:bg-transparent">
                <TableHead className="w-[52px] whitespace-nowrap">Pos</TableHead>
                <TableHead className={cn(teamColumnClass, 'whitespace-nowrap')}>Team</TableHead>
                <TableHead className="w-[48px] whitespace-nowrap text-right">P</TableHead>
                <TableHead className="w-[62px] whitespace-nowrap text-right">{isCurrentMode ? 'Pts' : 'Pred'}</TableHead>
                {showDeltaColumn && <TableHead className="w-[88px] whitespace-nowrap text-right">+/- vs Real</TableHead>}
                <TableHead className={cn(formColumnClass, 'whitespace-nowrap')}>Form (Last 5)</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => {
                const isFavorite = row.team === favoriteTeam
                return (
                  <TableRow key={`${mode}-${row.team}`} className={cn('h-14', isFavorite && tone.favorite)}>
                    <TableCell className={cn('w-14 font-semibold', tone.pos)}>{row.position}</TableCell>
                    <TableCell className={cn(teamColumnClass, 'font-medium')}>
                      <div className="flex items-center gap-2">
                        <ClubLogo team={row.team} />
                        <span className="whitespace-nowrap">{row.team}</span>
                      </div>
                    </TableCell>
                    <TableCell className="w-[48px] text-right tabular-nums">{row.played}</TableCell>
                    <TableCell className="w-[62px] text-right font-semibold tabular-nums">
                      {isCurrentMode ? row.points : row.predictedPoints}
                    </TableCell>
                    {showDeltaColumn && (
                      <TableCell className="w-[88px] text-right">
                        <span
                          className={cn(
                            'inline-flex min-w-[54px] justify-center rounded-full border px-2.5 py-1 text-xs font-semibold tabular-nums',
                            deltaClass(row.delta)
                          )}
                        >
                          {row.delta > 0 ? `+${row.delta}` : row.delta}
                        </span>
                      </TableCell>
                    )}
                    <TableCell className={formColumnClass}>
                      <FormChips results={isCurrentMode ? row.form : row.predictedForm} />
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  )
}

function PerformerList({ title, subtitle, items, favoriteTeam, direction }) {
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

function FinalModelTableCard({ rows, favoriteTeam, season }) {
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

function FavoriteTeamCard({ favoriteTeam, onFavoriteTeamChange, favoriteSnapshot, teamOptions }) {
  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-4">
        <CardTitle className="text-lg">Prediction Snapshot</CardTitle>
        <CardDescription>Track one club and compare current output against the projected model view.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 pt-0">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Favorite Club</p>
          <Select value={favoriteTeam} onValueChange={onFavoriteTeamChange}>
            <SelectTrigger className="border-slate-300 bg-white text-slate-900">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {teamOptions.map((team) => (
                <SelectItem key={team} value={team}>
                  <div className="flex items-center gap-2">
                    <ClubLogo team={team} />
                    <span>{team}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {favoriteSnapshot && (
          <div className="grid grid-cols-3 gap-2 rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div>
              <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Current</p>
              <p className="mt-1 text-sm font-semibold tabular-nums">{favoriteSnapshot.points} pts</p>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Predicted</p>
              <p className="mt-1 text-sm font-semibold tabular-nums">{favoriteSnapshot.predictedPoints} pts</p>
            </div>
            <div>
              <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Delta</p>
              <p className="mt-1 text-sm font-semibold tabular-nums">
                {favoriteSnapshot.delta > 0 ? '+' : ''}
                {favoriteSnapshot.delta}
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function OverviewPage({
  season,
  seasonOptions,
  onSeasonChange,
  topOver,
  topUnder,
  favoriteTeam,
  favoriteSnapshot,
  onFavoriteTeamChange,
  teamOptions,
}) {
  return (
    <div className="space-y-6">
      <div className="flex justify-end">
        <SeasonSelector season={season} seasonOptions={seasonOptions} onSeasonChange={onSeasonChange} />
      </div>
      <section className="grid items-start gap-6 lg:grid-cols-[1.35fr_0.9fr]">
        <div className="space-y-5">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Premier League Predictor</p>
          <h1 className="max-w-4xl text-4xl font-semibold leading-tight tracking-tight md:text-5xl">
            Smarter Premier League Predictions, Backed by Data, Not Guesswork
          </h1>
          <p className="max-w-3xl text-lg leading-relaxed text-slate-700">
            Match-by-match insights, team form analysis, and AI-driven predictions to help you stay ahead every
            gameweek.
          </p>
          <p className="max-w-3xl text-base leading-relaxed text-muted-foreground md:text-lg">
            This platform delivers data-driven predictions for every Premier League fixture. By combining statistical
            modelling, historical performance, and current team form, each forecast is designed to go beyond
            surface-level analysis and highlight the factors that genuinely influence match outcomes.
          </p>
          <div className="flex flex-wrap gap-2 pt-1">
            <Badge variant="outline" className="border-slate-200 bg-white text-slate-700">
              AI-driven forecasts
            </Badge>
            <Badge variant="outline" className="border-slate-200 bg-white text-slate-700">
              Match-by-match analysis
            </Badge>
            <Badge variant="outline" className="border-slate-200 bg-white text-slate-700">
              Structured model insights
            </Badge>
          </div>
        </div>
        <FavoriteTeamCard
          favoriteTeam={favoriteTeam}
          onFavoriteTeamChange={onFavoriteTeamChange}
          favoriteSnapshot={favoriteSnapshot}
          teamOptions={teamOptions}
        />
      </section>

      <section className="grid gap-6 md:grid-cols-2">
        <PerformerList
          title="Biggest over-performers"
          subtitle="Points above predicted"
          items={topOver}
          favoriteTeam={favoriteTeam}
          direction="up"
        />
        <PerformerList
          title="Biggest under-performers"
          subtitle="Points below predicted"
          items={topUnder}
          favoriteTeam={favoriteTeam}
          direction="down"
        />
      </section>
    </div>
  )
}

function TablesPage({
  season,
  seasonOptions,
  onSeasonChange,
  currentTable,
  predictedTable,
  favoriteTeam,
  gameweek,
  onGameweekChange,
}) {
  return (
    <div className="space-y-4">
      <section className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Standings</p>
          <h2 className="text-3xl font-semibold tracking-tight text-slate-900">Current vs Predicted Tables</h2>
          <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
            Compare live and projected positions by gameweek without crowding the rest of the dashboard.
          </p>
        </div>
        <SeasonSelector season={season} seasonOptions={seasonOptions} onSeasonChange={onSeasonChange} />
      </section>
      <section className="grid items-stretch gap-4 xl:grid-cols-2">
        <TableCard
          title="Current table"
          rows={currentTable}
          favoriteTeam={favoriteTeam}
          mode="current"
          gameweek={gameweek}
          onGameweekChange={onGameweekChange}
        />
        <TableCard
          title="Predicted table"
          rows={predictedTable}
          favoriteTeam={favoriteTeam}
          mode="predicted"
          gameweek={gameweek}
          onGameweekChange={onGameweekChange}
        />
      </section>
    </div>
  )
}

function MatchInsightCard({ title, description, match, tone = 'neutral', icon: Icon = Activity }) {
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
              Model: {matchOutcomeLabelMap[match.modelPickCode]} ({formatPercent(match.modelConfidence)})
            </p>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">No matching fixtures available.</p>
        )}
      </CardContent>
    </Card>
  )
}

function average(values) {
  const finiteValues = values.filter(Number.isFinite)
  if (!finiteValues.length) return null
  return finiteValues.reduce((sum, value) => sum + value, 0) / finiteValues.length
}

function sumFinite(values) {
  return values.reduce((sum, value) => sum + (Number.isFinite(value) ? value : 0), 0)
}

function buildOutcomeMix(matches, pickKey = 'fullTimeResult') {
  const total = Math.max(matches.length, 1)
  return [
    { code: 'H', label: 'Home Win', count: matches.filter((match) => match[pickKey] === 'H').length },
    { code: 'D', label: 'Draw', count: matches.filter((match) => match[pickKey] === 'D').length },
    { code: 'A', label: 'Away Win', count: matches.filter((match) => match[pickKey] === 'A').length },
  ].map((row) => ({ ...row, share: row.count / total }))
}

function buildMatchPageInsightData(matches) {
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

function MatchSummaryTiles({ summary }) {
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

function OutcomeMixChart({ title, rows }) {
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

function GameTexturePanel({ texture }) {
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

function TeamSignalPanel({ teams }) {
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

function ModelEdgePanel({ matches, onSelectMatch }) {
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

function safeChartValue(value) {
  return Number.isFinite(value) ? value : 0
}

function ratioOrNull(part, whole) {
  if (!Number.isFinite(part) || !Number.isFinite(whole) || whole <= 0) return null
  return part / whole
}

function barPercent(value, maxValue = 1) {
  const safeValue = Math.max(0, safeChartValue(value))
  const safeMax = Math.max(1, safeChartValue(maxValue))
  if (safeValue <= 0) return '0%'
  return `${Math.max((safeValue / safeMax) * 100, 3)}%`
}

function probabilityBarHeight(value) {
  const safeValue = Math.max(0, Math.min(1, safeChartValue(value)))
  if (safeValue <= 0) return '0%'
  return `${Math.max(safeValue * 100, 3)}%`
}

function outcomeProbabilityRows(match) {
  return [
    {
      code: 'H',
      label: 'Home Win',
      shortLabel: 'Home',
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
      label: 'Away Win',
      shortLabel: 'Away',
      team: match.awayTeam,
      model: match.modelAwayProb,
      market: match.marketAwayProb,
      odds: match.b365AwayOdds,
    },
  ]
}

function MatchOverviewPanel({ match }) {
  const marketPickCode = deriveMarketPickCode(match)
  const halfTimeScore = formatScoreline(match.halfTimeHomeGoals, match.halfTimeAwayGoals)
  const predictionStatus = match.predictionCorrect ? 'Correct' : 'Miss'

  const overviewItems = [
    { label: 'Date', value: match.matchDate },
    { label: 'Full Time', value: formatScoreline(match.homeGoals, match.awayGoals), emphasis: true },
    { label: 'Half Time', value: halfTimeScore },
    { label: 'Model Pick', value: matchOutcomeLabelMap[match.modelPickCode] ?? 'Unknown' },
    { label: 'Market Pick', value: matchOutcomeLabelMap[marketPickCode] ?? 'Unknown' },
    { label: 'Confidence', value: formatPercent(match.modelConfidence), emphasis: true },
  ]

  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardContent className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-7">
        {overviewItems.map((item) => (
          <div key={item.label} className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">{item.label}</p>
            <p className={cn('mt-1 font-semibold text-slate-900', item.emphasis ? 'text-xl tabular-nums' : 'text-sm')}>
              {item.value}
            </p>
          </div>
        ))}
        <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Actual Result</p>
          <Badge variant="outline" className={cn('mt-2 uppercase tracking-[0.12em]', matchOutcomeBadgeClass(match.fullTimeResult))}>
            {matchOutcomeLabelMap[match.fullTimeResult] ?? 'Unknown'}
          </Badge>
        </div>
        <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
          <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Prediction</p>
          <Badge
            variant="outline"
            className={cn(
              'mt-2 uppercase tracking-[0.12em]',
              match.predictionCorrect
                ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                : 'border-rose-200 bg-rose-50 text-rose-700'
            )}
          >
            {predictionStatus}
          </Badge>
        </div>
      </CardContent>
    </Card>
  )
}

function ProbabilityComparisonChart({ match }) {
  const rows = outcomeProbabilityRows(match)
  const marketPickCode = deriveMarketPickCode(match)
  const largestModelEdge = rows
    .map((row) => ({ ...row, delta: row.model - row.market }))
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))[0]

  return (
    <Card className="h-full border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <CardTitle className="text-base">Probability Comparison</CardTitle>
            <CardDescription>Model probabilities against the implied Bet365 market view.</CardDescription>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className={cn('uppercase tracking-[0.12em]', matchOutcomeBadgeClass(match.modelPickCode))}>
              Model: {matchOutcomeLabelMap[match.modelPickCode]}
            </Badge>
            <Badge variant="outline" className={cn('uppercase tracking-[0.12em]', matchOutcomeBadgeClass(marketPickCode))}>
              Market: {matchOutcomeLabelMap[marketPickCode]}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
        <div className="rounded-lg border border-slate-200 bg-slate-50/60 p-4">
          <div className="flex h-56 items-end gap-4 border-b border-slate-200 pb-3">
            {rows.map((row) => (
              <div key={`probability-${row.code}`} className="flex min-w-0 flex-1 flex-col items-center gap-2">
                <div className="flex h-40 w-full items-end justify-center gap-2">
                  <div className="flex h-full w-8 items-end rounded-t bg-slate-200/80">
                    <div
                      className="w-full rounded-t bg-sky-500"
                      style={{ height: probabilityBarHeight(row.model) }}
                      title={`Model ${formatPercent(row.model)}`}
                    />
                  </div>
                  <div className="flex h-full w-8 items-end rounded-t bg-slate-200/80">
                    <div
                      className="w-full rounded-t bg-amber-500"
                      style={{ height: probabilityBarHeight(row.market) }}
                      title={`Market ${formatPercent(row.market)}`}
                    />
                  </div>
                </div>
                <div className="min-w-0 text-center">
                  <p className="truncate text-xs font-semibold text-slate-800">{row.shortLabel}</p>
                  <p className="text-[11px] text-slate-500">{row.code === match.fullTimeResult ? 'Actual' : 'Outcome'}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-3 flex flex-wrap items-center gap-3 text-xs text-slate-600">
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-sky-500" />
              Model
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-amber-500" />
              Market
            </span>
          </div>
        </div>

        <div className="space-y-3">
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Largest Model Edge</p>
            <p className="mt-1 text-lg font-semibold text-slate-900">{largestModelEdge.label}</p>
            <p className="text-sm text-slate-600">
              {formatSigned(largestModelEdge.delta * 100, 1)} pts versus market
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Odds Coverage</p>
            <p className="mt-1 text-lg font-semibold text-slate-900">{match.bookmakerOdds?.length ?? 0} books</p>
            <p className="text-sm text-slate-600">Opening and closing prices are listed below.</p>
          </div>
          <div className="space-y-2">
            {rows.map((row) => (
              <div key={`probability-row-${row.code}`} className="grid grid-cols-[4.5rem_1fr_auto] items-center gap-2 text-xs">
                <span className="truncate font-medium text-slate-700">{row.shortLabel}</span>
                <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full rounded-full bg-sky-500" style={{ width: probabilityBarHeight(row.model) }} />
                </div>
                <span className="tabular-nums text-slate-600">{formatPercent(row.model)}</span>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function ShotQualityChart({ match }) {
  const teams = [
    {
      team: match.homeTeam,
      goals: match.homeGoals,
      shots: match.homeShots,
      shotsOnTarget: match.homeShotsOnTarget,
      corners: match.homeCorners,
      color: 'bg-sky-500',
      light: 'bg-sky-100',
    },
    {
      team: match.awayTeam,
      goals: match.awayGoals,
      shots: match.awayShots,
      shotsOnTarget: match.awayShotsOnTarget,
      corners: match.awayCorners,
      color: 'bg-rose-500',
      light: 'bg-rose-100',
    },
  ]
  const maxVolume = Math.max(...teams.flatMap((team) => [safeChartValue(team.shots), safeChartValue(team.shotsOnTarget), safeChartValue(team.goals)]), 1)

  return (
    <Card className="h-full border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Shot Quality Funnel</CardTitle>
        <CardDescription>Volume, accuracy, and conversion from the available post-match stats.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {teams.map((team) => {
          const targetRate = ratioOrNull(team.shotsOnTarget, team.shots)
          const conversionRate = ratioOrNull(team.goals, team.shots)

          return (
            <div key={`shot-quality-${team.team}`} className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex min-w-0 items-center gap-2">
                  <ClubLogo team={team.team} />
                  <p className="truncate text-sm font-semibold text-slate-900">{team.team}</p>
                </div>
                <p className="text-lg font-semibold tabular-nums text-slate-900">{formatOptionalStat(team.goals)}</p>
              </div>
              <div className="mt-3 space-y-2">
                {[
                  ['Shots', team.shots],
                  ['On Target', team.shotsOnTarget],
                  ['Goals', team.goals],
                ].map(([label, value]) => (
                  <div key={`${team.team}-${label}`} className="grid grid-cols-[5rem_1fr_2.5rem] items-center gap-2 text-xs">
                    <span className="font-medium text-slate-600">{label}</span>
                    <div className={cn('h-2 overflow-hidden rounded-full', team.light)}>
                      <div className={cn('h-full rounded-full', team.color)} style={{ width: barPercent(value, maxVolume) }} />
                    </div>
                    <span className="text-right tabular-nums text-slate-700">{formatOptionalStat(value)}</span>
                  </div>
                ))}
              </div>
              <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
                <div className="rounded-md border border-slate-200 bg-white p-2">
                  <p className="text-slate-500">Target Rate</p>
                  <p className="mt-1 font-semibold tabular-nums text-slate-900">{formatOptionalPercent(targetRate)}</p>
                </div>
                <div className="rounded-md border border-slate-200 bg-white p-2">
                  <p className="text-slate-500">Conversion</p>
                  <p className="mt-1 font-semibold tabular-nums text-slate-900">{formatOptionalPercent(conversionRate)}</p>
                </div>
                <div className="rounded-md border border-slate-200 bg-white p-2">
                  <p className="text-slate-500">Corners</p>
                  <p className="mt-1 font-semibold tabular-nums text-slate-900">{formatOptionalStat(team.corners)}</p>
                </div>
              </div>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

function MatchShareChart({ match }) {
  const metrics = [
    { label: 'Goals', home: match.homeGoals, away: match.awayGoals },
    { label: 'Shots', home: match.homeShots, away: match.awayShots },
    { label: 'Shots On Target', home: match.homeShotsOnTarget, away: match.awayShotsOnTarget },
    { label: 'Corners', home: match.homeCorners, away: match.awayCorners },
    { label: 'Fouls', home: match.homeFouls, away: match.awayFouls },
    { label: 'Yellow Cards', home: match.homeYellowCards, away: match.awayYellowCards },
    { label: 'Red Cards', home: match.homeRedCards, away: match.awayRedCards },
  ].filter((metric) => Number.isFinite(metric.home) && Number.isFinite(metric.away))

  return (
    <Card className="h-full border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Match Share</CardTitle>
        <CardDescription>Side-by-side share of the main event and discipline stats.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
          <span className="truncate text-sky-700">{match.homeTeam}</span>
          <span>Metric</span>
          <span className="truncate text-right text-rose-700">{match.awayTeam}</span>
        </div>
        {metrics.map((metric) => {
          const total = safeChartValue(metric.home) + safeChartValue(metric.away)
          const homeShare = total > 0 ? (metric.home / total) * 100 : 50
          const awayShare = total > 0 ? 100 - homeShare : 50
          const delta = metric.home - metric.away

          return (
            <div key={`match-share-${metric.label}`} className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
              <div className="mb-2 flex items-center justify-between gap-3 text-xs">
                <span className="font-semibold tabular-nums text-slate-900">{metric.home}</span>
                <span className="text-center font-semibold text-slate-600">{metric.label}</span>
                <span className="font-semibold tabular-nums text-slate-900">{metric.away}</span>
              </div>
              <div className="flex h-3 overflow-hidden rounded-full bg-slate-100">
                <div className="bg-sky-500" style={{ width: `${homeShare}%` }} />
                <div className="bg-rose-500" style={{ width: `${awayShare}%` }} />
              </div>
              <p className="mt-2 text-center text-[11px] text-slate-500">
                Differential: {formatSigned(delta, 0)}
              </p>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

function MatchRadarChart({ match }) {
  const axes = [
    { label: 'Goals', home: safeChartValue(match.homeGoals), away: safeChartValue(match.awayGoals) },
    { label: 'Shots', home: safeChartValue(match.homeShots), away: safeChartValue(match.awayShots) },
    { label: 'On Target', home: safeChartValue(match.homeShotsOnTarget), away: safeChartValue(match.awayShotsOnTarget) },
    { label: 'Corners', home: safeChartValue(match.homeCorners), away: safeChartValue(match.awayCorners) },
    {
      label: 'Accuracy',
      home: safeChartValue(ratioOrNull(match.homeShotsOnTarget, match.homeShots)),
      away: safeChartValue(ratioOrNull(match.awayShotsOnTarget, match.awayShots)),
      scale: 1,
    },
  ]
  const center = 120
  const radius = 78
  const pointFor = (index, value) => {
    const axis = axes[index]
    const maxValue = axis.scale ?? Math.max(axis.home, axis.away, 1)
    const normalized = maxValue > 0 ? Math.min(value / maxValue, 1) : 0
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / axes.length
    return {
      x: center + Math.cos(angle) * radius * normalized,
      y: center + Math.sin(angle) * radius * normalized,
    }
  }
  const axisPoint = (index, distance = radius) => {
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / axes.length
    return {
      x: center + Math.cos(angle) * distance,
      y: center + Math.sin(angle) * distance,
    }
  }
  const homePoints = axes.map((axis, index) => pointFor(index, axis.home)).map((point) => `${point.x},${point.y}`).join(' ')
  const awayPoints = axes.map((axis, index) => pointFor(index, axis.away)).map((point) => `${point.x},${point.y}`).join(' ')

  return (
    <Card className="h-full border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Pressure Profile</CardTitle>
        <CardDescription>Normalized comparison across scoring, shot volume, accuracy, and set-piece pressure.</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 md:grid-cols-[15rem_1fr]">
        <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
          <svg viewBox="0 0 240 240" role="img" aria-label="Pressure profile radar chart" className="h-60 w-full">
            {[0.33, 0.66, 1].map((ring) => (
              <polygon
                key={`radar-ring-${ring}`}
                points={axes.map((_, index) => {
                  const point = axisPoint(index, radius * ring)
                  return `${point.x},${point.y}`
                }).join(' ')}
                fill="none"
                stroke="#cbd5e1"
                strokeWidth="1"
              />
            ))}
            {axes.map((axis, index) => {
              const end = axisPoint(index)
              const labelPoint = axisPoint(index, radius + 21)
              return (
                <g key={`radar-axis-${axis.label}`}>
                  <line x1={center} y1={center} x2={end.x} y2={end.y} stroke="#cbd5e1" strokeWidth="1" />
                  <text
                    x={labelPoint.x}
                    y={labelPoint.y}
                    textAnchor={Math.abs(labelPoint.x - center) < 10 ? 'middle' : labelPoint.x > center ? 'start' : 'end'}
                    dominantBaseline="middle"
                    className="fill-slate-500 text-[10px] font-semibold"
                  >
                    {axis.label}
                  </text>
                </g>
              )
            })}
            <polygon points={homePoints} fill="rgba(14, 165, 233, 0.24)" stroke="#0ea5e9" strokeWidth="2" />
            <polygon points={awayPoints} fill="rgba(244, 63, 94, 0.2)" stroke="#f43f5e" strokeWidth="2" />
          </svg>
          <div className="flex flex-wrap justify-center gap-3 text-xs text-slate-600">
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-sky-500" />
              {match.homeTeam}
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-rose-500" />
              {match.awayTeam}
            </span>
          </div>
        </div>
        <div className="grid gap-2 sm:grid-cols-2 md:grid-cols-1">
          {axes.map((axis) => (
            <div key={`radar-value-${axis.label}`} className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">{axis.label}</p>
              <div className="mt-2 flex items-center justify-between gap-3 text-sm font-semibold text-slate-900">
                <span className="truncate text-sky-700">
                  {axis.label === 'Accuracy' ? formatOptionalPercent(axis.home) : formatOptionalStat(axis.home)}
                </span>
                <span className="text-slate-400">vs</span>
                <span className="truncate text-right text-rose-700">
                  {axis.label === 'Accuracy' ? formatOptionalPercent(axis.away) : formatOptionalStat(axis.away)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function ScoreProgressionChart({ match }) {
  const hasHalfTime = Number.isFinite(match.halfTimeHomeGoals) && Number.isFinite(match.halfTimeAwayGoals)
  const phases = hasHalfTime
    ? [
        { label: 'Kickoff', home: 0, away: 0 },
        { label: 'Half Time', home: match.halfTimeHomeGoals, away: match.halfTimeAwayGoals },
        { label: 'Full Time', home: match.homeGoals, away: match.awayGoals },
      ]
    : [
        { label: 'Kickoff', home: 0, away: 0 },
        { label: 'Full Time', home: match.homeGoals, away: match.awayGoals },
      ]
  const maxGoals = Math.max(...phases.flatMap((phase) => [safeChartValue(phase.home), safeChartValue(phase.away)]), 1)
  const width = 320
  const height = 150
  const padding = 24
  const chartWidth = width - padding * 2
  const chartHeight = height - padding * 2
  const point = (phase, index) => {
    const x = padding + (index / Math.max(phases.length - 1, 1)) * chartWidth
    const homeY = padding + chartHeight - (safeChartValue(phase.home) / maxGoals) * chartHeight
    const awayY = padding + chartHeight - (safeChartValue(phase.away) / maxGoals) * chartHeight
    return { x, homeY, awayY }
  }
  const plotted = phases.map(point)
  const homeSecondHalf = hasHalfTime && Number.isFinite(match.homeGoals) ? match.homeGoals - match.halfTimeHomeGoals : null
  const awaySecondHalf = hasHalfTime && Number.isFinite(match.awayGoals) ? match.awayGoals - match.halfTimeAwayGoals : null

  return (
    <Card className="h-full border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Scoring Progression</CardTitle>
        <CardDescription>Half-time and full-time score movement where the fixture has interval data.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
          <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Score progression chart" className="h-44 w-full">
            <line x1={padding} y1={padding + chartHeight} x2={padding + chartWidth} y2={padding + chartHeight} stroke="#cbd5e1" />
            {[...Array(maxGoals + 1)].map((_, value) => {
              const y = padding + chartHeight - (value / maxGoals) * chartHeight
              return (
                <g key={`score-grid-${value}`}>
                  <line x1={padding} y1={y} x2={padding + chartWidth} y2={y} stroke="#e2e8f0" strokeWidth="1" />
                  <text x={8} y={y + 3} className="fill-slate-400 text-[10px]">
                    {value}
                  </text>
                </g>
              )
            })}
            <polyline
              points={plotted.map((plot) => `${plot.x},${plot.homeY}`).join(' ')}
              fill="none"
              stroke="#0ea5e9"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            <polyline
              points={plotted.map((plot) => `${plot.x},${plot.awayY}`).join(' ')}
              fill="none"
              stroke="#f43f5e"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            {plotted.map((plot, index) => (
              <g key={`score-point-${phases[index].label}`}>
                <circle cx={plot.x} cy={plot.homeY} r="4" fill="#0ea5e9" />
                <circle cx={plot.x} cy={plot.awayY} r="4" fill="#f43f5e" />
                <text x={plot.x} y={height - 5} textAnchor="middle" className="fill-slate-500 text-[10px] font-semibold">
                  {phases[index].label}
                </text>
              </g>
            ))}
          </svg>
          <div className="mt-2 flex flex-wrap justify-center gap-3 text-xs text-slate-600">
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-sky-500" />
              {match.homeTeam}
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="h-2.5 w-2.5 rounded-sm bg-rose-500" />
              {match.awayTeam}
            </span>
          </div>
        </div>
        <div className="grid gap-2 sm:grid-cols-3">
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Half Time</p>
            <p className="mt-1 text-lg font-semibold tabular-nums text-slate-900">
              {formatScoreline(match.halfTimeHomeGoals, match.halfTimeAwayGoals)}
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Second Half</p>
            <p className="mt-1 text-lg font-semibold tabular-nums text-slate-900">
              {formatScoreline(homeSecondHalf, awaySecondHalf)}
            </p>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Full Time</p>
            <p className="mt-1 text-lg font-semibold tabular-nums text-slate-900">
              {formatScoreline(match.homeGoals, match.awayGoals)}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function BookmakerOddsPanel({ match }) {
  const bookmakerOdds = Array.isArray(match.bookmakerOdds) ? match.bookmakerOdds : []

  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <div className="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <CardTitle className="text-base">Bookmaker Odds</CardTitle>
            <CardDescription>
              Opening and closing 1X2 odds from every bookmaker column available for this fixture.
            </CardDescription>
          </div>
          <Badge variant="outline" className="w-fit border-slate-200 bg-slate-50 text-slate-700">
            {bookmakerOdds.length} sources
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        {bookmakerOdds.length ? (
          <div className="matches-scroll-container overflow-x-auto rounded-lg border border-slate-200">
            <table className="w-full min-w-[860px] text-left text-sm">
              <thead className="bg-slate-50 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">
                <tr>
                  <th className="px-3 py-3">Bookmaker</th>
                  <th className="px-3 py-3 text-right">Open Home</th>
                  <th className="px-3 py-3 text-right">Open Draw</th>
                  <th className="px-3 py-3 text-right">Open Away</th>
                  <th className="px-3 py-3 text-right">Close Home</th>
                  <th className="px-3 py-3 text-right">Close Draw</th>
                  <th className="px-3 py-3 text-right">Close Away</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 bg-white">
                {bookmakerOdds.map((bookmaker) => (
                  <tr key={`bookmaker-odds-${bookmaker.code}`}>
                    <td className="px-3 py-3 font-semibold text-slate-900">
                      <span>{bookmaker.label}</span>
                      <span className="ml-2 text-xs font-medium text-slate-400">{bookmaker.code}</span>
                    </td>
                    <td className="px-3 py-3 text-right tabular-nums text-slate-700">{formatOdds(bookmaker.home)}</td>
                    <td className="px-3 py-3 text-right tabular-nums text-slate-700">{formatOdds(bookmaker.draw)}</td>
                    <td className="px-3 py-3 text-right tabular-nums text-slate-700">{formatOdds(bookmaker.away)}</td>
                    <td className="px-3 py-3 text-right tabular-nums text-slate-900">{formatOdds(bookmaker.closingHome)}</td>
                    <td className="px-3 py-3 text-right tabular-nums text-slate-900">{formatOdds(bookmaker.closingDraw)}</td>
                    <td className="px-3 py-3 text-right tabular-nums text-slate-900">{formatOdds(bookmaker.closingAway)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-4 text-sm text-slate-600">
            No bookmaker odds are available for this fixture.
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function MatchDetailsDrawer({ matches, activeIndex, onClose, onSelectIndex }) {
  if (activeIndex < 0 || activeIndex >= matches.length) return null

  const match = matches[activeIndex]
  const hasPrevious = activeIndex > 0
  const hasNext = activeIndex < matches.length - 1

  const drawer = (
    <div className="fixed inset-0 z-[100]">
      <button
        type="button"
        className="absolute inset-0 bg-slate-900/55"
        aria-label="Close details"
        onClick={onClose}
      />
      <aside className="absolute inset-2 overflow-hidden rounded-xl border border-slate-200 bg-slate-50 shadow-2xl sm:inset-4 lg:inset-6">
        <div className="flex h-full flex-col">
          <div className="flex flex-col gap-4 border-b border-slate-200 bg-white px-4 py-4 sm:px-5 lg:flex-row lg:items-center lg:justify-between lg:px-6">
            <div className="min-w-0">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Match Analysis</p>
              <div className="mt-2 flex min-w-0 flex-wrap items-center gap-3">
                <ClubLogo team={match.homeTeam} size="lg" />
                <div className="min-w-0">
                  <p className="truncate text-xl font-semibold tracking-tight text-slate-900 sm:text-2xl">
                    {match.homeTeam} vs {match.awayTeam}
                  </p>
                  <p className="mt-1 text-sm text-slate-600">
                    {match.matchDate} · {formatScoreline(match.homeGoals, match.awayGoals)}
                  </p>
                </div>
                <ClubLogo team={match.awayTeam} size="lg" />
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-300 bg-white text-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
                onClick={() => hasPrevious && onSelectIndex(activeIndex - 1)}
                disabled={!hasPrevious}
                aria-label="Previous match"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-300 bg-white text-slate-700 disabled:cursor-not-allowed disabled:opacity-40"
                onClick={() => hasNext && onSelectIndex(activeIndex + 1)}
                disabled={!hasNext}
                aria-label="Next match"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-slate-300 bg-white text-slate-700"
                onClick={onClose}
                aria-label="Close drawer"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div className="drawer-scroll flex-1 overflow-y-auto p-4 sm:p-5 lg:p-6">
            <div className="mx-auto max-w-[1500px] space-y-4">
              <MatchOverviewPanel match={match} />
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_minmax(360px,0.75fr)]">
                <ProbabilityComparisonChart match={match} />
                <ScoreProgressionChart match={match} />
              </div>
              <BookmakerOddsPanel match={match} />
              <div className="grid gap-4 xl:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
                <ShotQualityChart match={match} />
                <MatchRadarChart match={match} />
              </div>
              <MatchShareChart match={match} />
            </div>
          </div>
        </div>
      </aside>
    </div>
  )

  if (typeof document === 'undefined') return drawer
  return createPortal(drawer, document.body)
}

function MatchColumnsMenu({ columnVisibility, onToggleColumn, onResetColumns }) {
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    if (!isOpen) return undefined
    const handlePointerDown = (event) => {
      const target = event.target
      if (!(target instanceof HTMLElement)) return
      if (!target.closest('[data-columns-menu-root]')) {
        setIsOpen(false)
      }
    }
    window.addEventListener('pointerdown', handlePointerDown)
    return () => window.removeEventListener('pointerdown', handlePointerDown)
  }, [isOpen])

  return (
    <div className="relative" data-columns-menu-root>
      <button
        type="button"
        className="inline-flex h-10 items-center gap-2 rounded-md border border-slate-300 bg-white px-3 text-sm font-semibold text-slate-700 hover:border-slate-400"
        onClick={() => setIsOpen((current) => !current)}
      >
        <Columns3 className="h-4 w-4" />
        Columns
      </button>
      {isOpen && (
        <div className="absolute right-0 z-20 mt-2 w-64 rounded-lg border border-slate-200 bg-white p-3 shadow-xl">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Visible Columns</p>
            <button
              type="button"
              className="text-xs font-semibold text-slate-600 underline-offset-2 hover:underline"
              onClick={onResetColumns}
            >
              Reset
            </button>
          </div>
          <div className="grid grid-cols-1 gap-1.5">
            {MATCH_COLUMN_DEFINITIONS.map((column) => (
              <label
                key={`match-column-${column.key}`}
                className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
              >
                <input
                  type="checkbox"
                  checked={columnVisibility[column.key] !== false}
                  onChange={() => onToggleColumn(column.key)}
                />
                <span>{column.label}</span>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function MatchesPage({
  season,
  seasonOptions,
  onSeasonChange,
  matches,
  insights,
  columnVisibility,
  onToggleColumn,
  onResetColumns,
}) {
  const [activeMatchIndex, setActiveMatchIndex] = useState(-1)

  useEffect(() => {
    if (!matches.length) {
      setActiveMatchIndex(-1)
      return
    }
    if (activeMatchIndex >= matches.length) {
      setActiveMatchIndex(matches.length - 1)
    }
  }, [matches, activeMatchIndex])

  useEffect(() => {
    if (activeMatchIndex < 0) return undefined
    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        setActiveMatchIndex(-1)
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [activeMatchIndex])

  const isVisible = (columnKey) => columnVisibility[columnKey] !== false

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Matches & Results</p>
          <h2 className="text-3xl font-semibold tracking-tight text-slate-900">All Fixtures, Scores, and Confidence</h2>
          <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
            Review every played fixture for the selected season with model confidence, probabilities, and key match stats.
          </p>
        </div>
        <SeasonSelector season={season} seasonOptions={seasonOptions} onSeasonChange={onSeasonChange} />
      </div>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MatchInsightCard
          title="Highest Confidence"
          description="Model's strongest pick"
          match={insights.highestConfidence}
          tone="positive"
          icon={TrendingUp}
        />
        <MatchInsightCard
          title="Lowest Confidence"
          description="Closest-call fixture"
          match={insights.lowestConfidence}
          tone="caution"
          icon={Activity}
        />
        <MatchInsightCard
          title="High-Confidence Miss"
          description="Biggest confident miss"
          match={insights.highestConfidenceMiss}
          tone="negative"
          icon={TrendingDown}
        />
        <MatchInsightCard
          title="Biggest Winning Margin"
          description="Largest scoreline gap"
          match={insights.biggestGoalMargin}
          tone="highlight"
          icon={Sparkles}
        />
      </section>

      <Card className="border-slate-200 bg-white shadow-sm">
        <CardContent className="pt-6">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-0.5">
              <p className="text-sm text-slate-600">{matches.length} fixtures</p>
              <p className="text-xs text-slate-500">Click any row to open the side detail panel.</p>
            </div>
            <MatchColumnsMenu
              columnVisibility={columnVisibility}
              onToggleColumn={onToggleColumn}
              onResetColumns={onResetColumns}
            />
          </div>

          <div className="matches-scroll-container overflow-x-auto overflow-y-hidden rounded-lg border border-slate-200">
            <Table className="min-w-[980px]">
              <TableHeader className="bg-slate-50">
                <TableRow className="border-b-0 hover:bg-transparent">
                  {isVisible('date') && <TableHead>Date</TableHead>}
                  {isVisible('home') && <TableHead>Home</TableHead>}
                  {isVisible('away') && <TableHead>Away</TableHead>}
                  {isVisible('score') && <TableHead className="text-right">Score</TableHead>}
                  {isVisible('result') && <TableHead>Result</TableHead>}
                  {isVisible('modelPick') && <TableHead>Model Pick</TableHead>}
                  {isVisible('confidence') && <TableHead>Confidence</TableHead>}
                  {isVisible('prediction') && <TableHead>Prediction</TableHead>}
                </TableRow>
              </TableHeader>
              <TableBody>
                {matches.map((match, index) => (
                  <TableRow
                    key={match.id}
                    className="cursor-pointer hover:bg-slate-50"
                    onClick={() => setActiveMatchIndex(index)}
                  >
                    {isVisible('date') && <TableCell className="font-medium tabular-nums">{match.matchDate}</TableCell>}
                    {isVisible('home') && (
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <ClubLogo team={match.homeTeam} />
                          <span>{match.homeTeam}</span>
                        </div>
                      </TableCell>
                    )}
                    {isVisible('away') && (
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <ClubLogo team={match.awayTeam} />
                          <span>{match.awayTeam}</span>
                        </div>
                      </TableCell>
                    )}
                    {isVisible('score') && (
                      <TableCell className="text-right font-semibold tabular-nums">
                        {formatScoreline(match.homeGoals, match.awayGoals)}
                      </TableCell>
                    )}
                    {isVisible('result') && (
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={cn('uppercase tracking-[0.12em]', matchOutcomeBadgeClass(match.fullTimeResult))}
                        >
                          {matchOutcomeLabelMap[match.fullTimeResult] ?? 'Unknown'}
                        </Badge>
                      </TableCell>
                    )}
                    {isVisible('modelPick') && (
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={cn('uppercase tracking-[0.12em]', matchOutcomeBadgeClass(match.modelPickCode))}
                        >
                          {matchOutcomeLabelMap[match.modelPickCode] ?? 'Unknown'}
                        </Badge>
                      </TableCell>
                    )}
                    {isVisible('confidence') && (
                      <TableCell>
                        <span
                          className={cn(
                            'inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold tabular-nums',
                            confidenceBadgeClass(match.modelConfidence)
                          )}
                        >
                          {formatPercent(match.modelConfidence)}
                        </span>
                      </TableCell>
                    )}
                    {isVisible('prediction') && (
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={cn(
                            'uppercase tracking-[0.12em]',
                            match.predictionCorrect
                              ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                              : 'border-rose-200 bg-rose-50 text-rose-700'
                          )}
                        >
                          {match.predictionCorrect ? 'Correct' : 'Miss'}
                        </Badge>
                      </TableCell>
                    )}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <MatchDetailsDrawer
        matches={matches}
        activeIndex={activeMatchIndex}
        onClose={() => setActiveMatchIndex(-1)}
        onSelectIndex={setActiveMatchIndex}
      />
    </section>
  )
}

function MethodologyPage() {
  return (
    <section className="space-y-4">
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">How It Works</p>
        <h2 className="text-3xl font-semibold tracking-tight text-slate-900">
          Every prediction is built on a structured approach.
        </h2>
        <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
          Rather than relying on guesswork, the platform focuses on measurable signals that shape results across a
          full Premier League season.
        </p>
      </div>
      <Card className="border-slate-200 bg-white shadow-sm">
        <CardContent className="grid gap-4 p-6 md:grid-cols-2 xl:grid-cols-4">
          {methodology.map((item) => {
            const Icon = item.icon
            return (
              <div key={item.title} className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-slate-200">
                  <Icon className="h-5 w-5 text-slate-700" />
                </div>
                <h3 className="text-base font-semibold text-slate-900">{item.title}</h3>
                <p className="mt-2 text-sm leading-6 text-slate-600">{item.description}</p>
              </div>
            )
          })}
        </CardContent>
      </Card>
    </section>
  )
}

function ModelOutputPage({ favoriteTeam, matches, modelOutputTable, season, seasonOptions, onSeasonChange }) {
  const [activeView, setActiveView] = useState('insights')
  const [activeMatchIndex, setActiveMatchIndex] = useState(-1)
  const modelInsights = useMemo(() => buildMatchPageInsightData(matches), [matches])

  useEffect(() => {
    if (!matches.length) {
      setActiveMatchIndex(-1)
      return
    }
    if (activeMatchIndex >= matches.length) {
      setActiveMatchIndex(matches.length - 1)
    }
  }, [matches, activeMatchIndex])

  useEffect(() => {
    if (activeMatchIndex < 0) return undefined
    const handleEscape = (event) => {
      if (event.key === 'Escape') {
        setActiveMatchIndex(-1)
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [activeMatchIndex])

  const openMatchById = (matchId) => {
    const nextIndex = matches.findIndex((match) => match.id === matchId)
    if (nextIndex >= 0) setActiveMatchIndex(nextIndex)
  }

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Model Output</p>
          <h2 className="text-3xl font-semibold tracking-tight text-slate-900">Model Insights and Predicted Table</h2>
          <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
            Review how the model compares with results and market prices, then switch to the predicted season table.
          </p>
        </div>
        <SeasonSelector season={season} seasonOptions={seasonOptions} onSeasonChange={onSeasonChange} />
      </div>

      <div className="inline-flex rounded-lg border border-slate-200 bg-white p-1">
        {[
          { key: 'insights', label: 'Model Insights' },
          { key: 'table', label: 'Predicted Table' },
        ].map((item) => (
          <button
            key={item.key}
            type="button"
            className={cn(
              'rounded-md px-3 py-2 text-sm font-semibold transition-colors',
              activeView === item.key
                ? 'bg-slate-900 text-white'
                : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
            )}
            onClick={() => setActiveView(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {activeView === 'insights' ? (
        <div className="space-y-4">
          <MatchSummaryTiles summary={modelInsights.summary} />

          <section className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(320px,0.9fr)]">
            <div className="grid gap-4 lg:grid-cols-3 xl:grid-cols-1">
              <OutcomeMixChart title="Actual Outcomes" rows={modelInsights.actualOutcomeMix} />
              <OutcomeMixChart title="Model Picks" rows={modelInsights.modelOutcomeMix} />
              <OutcomeMixChart title="Market Picks" rows={modelInsights.marketOutcomeMix} />
            </div>
            <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-1">
              <GameTexturePanel texture={modelInsights.gameTexture} />
              <ModelEdgePanel
                matches={modelInsights.modelEdges}
                onSelectMatch={(match) => openMatchById(match.id)}
              />
            </div>
          </section>

          <TeamSignalPanel teams={modelInsights.teamSignals} />
        </div>
      ) : (
        <FinalModelTableCard rows={modelOutputTable} favoriteTeam={favoriteTeam} season={season} />
      )}

      <MatchDetailsDrawer
        matches={matches}
        activeIndex={activeMatchIndex}
        onClose={() => setActiveMatchIndex(-1)}
        onSelectIndex={setActiveMatchIndex}
      />
    </section>
  )
}

function ClubPage({
  season,
  seasonOptions,
  onSeasonChange,
  clubs,
  selectedClub,
  onSelectedClubChange,
  clubFixtures,
  clubSummary,
}) {
  const [activeMatchIndex, setActiveMatchIndex] = useState(-1)

  useEffect(() => {
    if (!clubFixtures.length) {
      setActiveMatchIndex(-1)
      return
    }
    if (activeMatchIndex >= clubFixtures.length) {
      setActiveMatchIndex(clubFixtures.length - 1)
    }
  }, [clubFixtures, activeMatchIndex])

  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Club View</p>
          <h2 className="text-3xl font-semibold tracking-tight text-slate-900">
            Fixtures, Results, and Model Confidence
          </h2>
          <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
            Select any club to view all fixtures, match outcomes, and model confidence for the chosen season.
          </p>
        </div>
        <SeasonSelector season={season} seasonOptions={seasonOptions} onSeasonChange={onSeasonChange} />
      </div>

      <Card className="border-slate-200 bg-white shadow-sm">
        <CardHeader className="space-y-5 pb-4">
          <div>
            <CardTitle className="text-lg">Club Breakdown</CardTitle>
          </div>

          <div className="space-y-2">
            <p className="text-xs uppercase tracking-[0.16em] text-muted-foreground">Choose Club</p>
            <Select value={selectedClub} onValueChange={onSelectedClubChange}>
              <SelectTrigger className="h-16 w-full border border-slate-300 bg-white px-5 text-lg font-semibold text-slate-900 sm:h-20 sm:text-2xl">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="max-h-[420px]">
                {clubs.map((club) => (
                  <SelectItem key={club} value={club} className="py-3 text-lg">
                    <div className="flex items-center gap-3">
                      <ClubLogo team={club} size="lg" />
                      <span>{club}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardHeader>
        <CardContent className="space-y-4 pt-0">
          {clubSummary ? (
            <>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Played</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums">{clubSummary.played}</p>
                </div>
                <div className="rounded-lg border border-sky-200 bg-sky-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-sky-700">Points</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums text-sky-700">{clubSummary.actualPoints}</p>
                </div>
                <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-emerald-700">Wins</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums text-emerald-700">{clubSummary.wins}</p>
                </div>
                <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Draws</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums">{clubSummary.draws}</p>
                </div>
                <div className="rounded-lg border border-rose-200 bg-rose-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-rose-700">Losses</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums text-rose-700">{clubSummary.losses}</p>
                </div>
                <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-amber-700">Model Accuracy / Confidence</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums text-amber-700">
                    {formatPercent(clubSummary.modelAccuracy)} / {formatPercent(clubSummary.averageConfidence)}
                  </p>
                </div>
              </div>

              <div>
                <div className="space-y-1">
                  <p className="text-xs uppercase tracking-[0.16em] text-indigo-700">Actual vs Model Expectation</p>
                  <p className="text-sm text-indigo-900/90">Delta is actual minus expected from model probabilities.</p>
                </div>
                <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                  <div className="rounded-lg border border-indigo-200 bg-white p-3">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Wins</p>
                    <p className="mt-1 text-sm tabular-nums">
                      {clubSummary.wins} actual vs {clubSummary.expectedWins.toFixed(1)} expected
                    </p>
                    <p className={cn('mt-1 text-sm font-semibold tabular-nums', comparisonDeltaClass(clubSummary.winDelta))}>
                      {formatSigned(clubSummary.winDelta)}
                    </p>
                  </div>
                  <div className="rounded-lg border border-indigo-200 bg-white p-3">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Draws</p>
                    <p className="mt-1 text-sm tabular-nums">
                      {clubSummary.draws} actual vs {clubSummary.expectedDraws.toFixed(1)} expected
                    </p>
                    <p className={cn('mt-1 text-sm font-semibold tabular-nums', comparisonDeltaClass(clubSummary.drawDelta))}>
                      {formatSigned(clubSummary.drawDelta)}
                    </p>
                  </div>
                  <div className="rounded-lg border border-indigo-200 bg-white p-3">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Losses</p>
                    <p className="mt-1 text-sm tabular-nums">
                      {clubSummary.losses} actual vs {clubSummary.expectedLosses.toFixed(1)} expected
                    </p>
                    <p className={cn('mt-1 text-sm font-semibold tabular-nums', comparisonDeltaClass(clubSummary.lossDelta, true))}>
                      {formatSigned(clubSummary.lossDelta)}
                    </p>
                  </div>
                  <div className="rounded-lg border border-indigo-200 bg-white p-3">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Points</p>
                    <p className="mt-1 text-sm tabular-nums">
                      {clubSummary.actualPoints} actual vs {clubSummary.expectedPoints.toFixed(1)} expected
                    </p>
                    <p className={cn('mt-1 text-sm font-semibold tabular-nums', comparisonDeltaClass(clubSummary.pointDelta))}>
                      {formatSigned(clubSummary.pointDelta)}
                    </p>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">No fixtures available for this club in {season || 'this season'}.</p>
          )}

          <div className="overflow-x-auto overflow-y-hidden rounded-lg border border-slate-200">
            <div className="border-b border-slate-200 bg-slate-50 px-4 py-2">
              <p className="text-xs text-slate-500">Click any row to open the side detail panel.</p>
            </div>
            <Table className="min-w-[960px]">
              <TableHeader className="bg-slate-50">
                <TableRow className="border-b-0 hover:bg-transparent">
                  <TableHead>Date</TableHead>
                  <TableHead>Opponent</TableHead>
                  <TableHead>Venue</TableHead>
                  <TableHead>Result</TableHead>
                  <TableHead>Model Pick</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead className="text-right">Win %</TableHead>
                  <TableHead className="text-right">Draw %</TableHead>
                  <TableHead className="text-right">Loss %</TableHead>
                  <TableHead>Prediction</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {clubFixtures.map((fixture, index) => (
                  <TableRow
                    key={fixture.id}
                    className="h-14 cursor-pointer hover:bg-slate-50"
                    onClick={() => setActiveMatchIndex(index)}
                  >
                    <TableCell className="font-medium tabular-nums">{fixture.matchDate}</TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <ClubLogo team={fixture.opponent} />
                        <span>{fixture.opponent}</span>
                      </div>
                    </TableCell>
                    <TableCell>{fixture.venue}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className={cn('uppercase tracking-[0.12em]', outcomeBadgeClass(fixture.actualOutcome))}>
                        {outcomeLabelMap[fixture.actualOutcome]}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className={cn('uppercase tracking-[0.12em]', outcomeBadgeClass(fixture.modelOutcome))}>
                        {outcomeLabelMap[fixture.modelOutcome]}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <span
                        className={cn(
                          'inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold tabular-nums',
                          confidenceBadgeClass(fixture.modelConfidence)
                        )}
                      >
                        {formatPercent(fixture.modelConfidence)}
                      </span>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{formatPercent(fixture.winProbability)}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatPercent(fixture.drawProbability)}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatPercent(fixture.lossProbability)}</TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={cn(
                          'uppercase tracking-[0.12em]',
                          fixture.predictionCorrect
                            ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                            : 'border-slate-200 bg-slate-100 text-slate-700'
                        )}
                      >
                        {fixture.predictionCorrect ? 'Correct' : 'Miss'}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <MatchDetailsDrawer
        matches={clubFixtures}
        activeIndex={activeMatchIndex}
        onClose={() => setActiveMatchIndex(-1)}
        onSelectIndex={setActiveMatchIndex}
      />
    </section>
  )
}

function AppNavigation() {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/90 backdrop-blur-sm">
      <div className="mx-auto w-full max-w-[1320px] px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-3 py-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Premier League Predictor</p>
            <p className="text-lg font-semibold tracking-tight text-slate-900">Data-driven match insights</p>
          </div>
          <nav className="flex gap-2 overflow-x-auto pb-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  cn(
                    'rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.12em] transition-colors',
                    isActive
                      ? 'border-slate-900 bg-slate-900 text-white'
                      : 'border-slate-300 bg-white text-slate-700 hover:border-slate-400 hover:text-slate-900'
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </div>
    </header>
  )
}

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
      sortFixturesChronologically(seasonFixtures).map((match, index) => {
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
