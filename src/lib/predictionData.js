import { normalizeTeamName } from './teamUtils'

export function seasonStartFromLabel(seasonLabel) {
  const [startToken] = String(seasonLabel).split('/')
  const parsed = Number.parseInt(startToken, 10)
  return Number.isFinite(parsed) ? parsed : Number.NEGATIVE_INFINITY
}

export function inferSeasonFromMatchDate(dateValue) {
  const parsedDate = new Date(dateValue)
  if (Number.isNaN(parsedDate.getTime())) return ''
  const year = parsedDate.getUTCFullYear()
  const month = parsedDate.getUTCMonth() + 1
  const seasonStart = month >= 7 ? year : year - 1
  return `${seasonStart}/${String(seasonStart + 1).slice(-2)}`
}

export function parseNumber(value, fallback = 0) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

export function parseNullableNumber(value) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

export function fixtureLookupKey(matchDate, homeTeam, awayTeam) {
  return `${matchDate}::${normalizeTeamName(homeTeam)}::${normalizeTeamName(awayTeam)}`
}

export function normalizeFixtureDate(dateValue) {
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

export function bookmakerLabel(code) {
  return bookmakerLabelMap[code] ?? code
}

export function parseSupplementalMatchStats(rawCsv) {
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

export function collectBookmakerOdds(headers) {
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

export function readBookmakerOdds(values, readValue, bookmakerColumns) {
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

export function parseRawSeasonMatchDetails(rawCsvModules) {
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

export function fallbackBookmakerOddsFromPrediction(match) {
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

export function parsePredictionFixtures(rawCsv, supplementalCsv = '', rawOddsCsvModules = {}) {
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
