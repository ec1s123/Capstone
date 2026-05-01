// This code was generated with Codex.
const teamAliases = {
  'AFC Bournemouth': 'Bournemouth',
  Birmingham: 'Birmingham City',
  Blackburn: 'Blackburn Rovers',
  Blackpool: 'Blackpool',
  Bolton: 'Bolton Wanderers',
  Bradford: 'Bradford City',
  'Brighton & Hove Albion': 'Brighton',
  Cardiff: 'Cardiff City',
  Charlton: 'Charlton Athletic',
  Coventry: 'Coventry City',
  Derby: 'Derby County',
  Huddersfield: 'Huddersfield Town',
  Hull: 'Hull City',
  Luton: 'Luton Town',
  'Man City': 'Manchester City',
  'Man United': 'Manchester United',
  'Man Utd': 'Manchester United',
  Middlesbrough: 'Middlesbrough',
  Newcastle: 'Newcastle United',
  Norwich: 'Norwich City',
  "Nott'm Forest": 'Nottingham Forest',
  Portsmouth: 'Portsmouth',
  QPR: 'Queens Park Rangers',
  Reading: 'Reading',
  Stoke: 'Stoke City',
  Swansea: 'Swansea City',
  Spurs: 'Tottenham',
  'Tottenham Hotspur': 'Tottenham',
  'West Brom': 'West Bromwich Albion',
  'West Ham United': 'West Ham',
  Wigan: 'Wigan Athletic',
  Wolves: 'Wolverhampton Wanderers',
}

const teamDomains = {
  Arsenal: 'arsenal.com',
  'Aston Villa': 'avfc.co.uk',
  'Birmingham City': 'bcfc.com',
  'Blackburn Rovers': 'rovers.co.uk',
  Blackpool: 'blackpoolfc.co.uk',
  'Bolton Wanderers': 'bwfc.co.uk',
  Bournemouth: 'afcb.co.uk',
  'Bradford City': 'bradfordcityafc.com',
  Brentford: 'brentfordfc.com',
  Brighton: 'brightonandhovealbion.com',
  Burnley: 'burnleyfootballclub.com',
  'Cardiff City': 'cardiffcityfc.co.uk',
  'Charlton Athletic': 'charltonafc.com',
  Chelsea: 'chelseafc.com',
  'Coventry City': 'ccfc.co.uk',
  'Crystal Palace': 'cpfc.co.uk',
  'Derby County': 'dcfc.co.uk',
  Everton: 'evertonfc.com',
  Fulham: 'fulhamfc.com',
  'Huddersfield Town': 'htafc.com',
  'Hull City': 'wearehullcity.co.uk',
  Ipswich: 'itfc.co.uk',
  Leeds: 'leedsunited.com',
  Leicester: 'lcfc.com',
  Liverpool: 'liverpoolfc.com',
  'Luton Town': 'lutontown.co.uk',
  'Manchester City': 'mancity.com',
  'Manchester United': 'manutd.com',
  Middlesbrough: 'mfc.co.uk',
  'Newcastle United': 'nufc.co.uk',
  'Norwich City': 'canaries.co.uk',
  'Nottingham Forest': 'nottinghamforest.co.uk',
  Portsmouth: 'portsmouthfc.co.uk',
  Reading: 'readingfc.co.uk',
  'Queens Park Rangers': 'qpr.co.uk',
  Southampton: 'southamptonfc.com',
  'Sheffield United': 'sufc.co.uk',
  'Stoke City': 'stokecityfc.com',
  Sunderland: 'safc.com',
  'Swansea City': 'swanseacity.com',
  Tottenham: 'tottenhamhotspur.com',
  Watford: 'watfordfc.com',
  'West Bromwich Albion': 'wbafc.co.uk',
  'West Ham': 'whufc.com',
  'Wigan Athletic': 'wiganathletic.com',
  Wolves: 'wolves.co.uk',
  'Wolverhampton Wanderers': 'wolves.co.uk',
}

const teamDisplayNames = {
  'Manchester City': 'Man City',
  'Manchester United': 'Man United',
  'Newcastle United': 'Newcastle',
  'Nottingham Forest': "Nott'm Forest",
  'Queens Park Rangers': 'QPR',
  'Sheffield United': 'Sheffield Utd',
  'Tottenham Hotspur': 'Tottenham',
  'West Bromwich Albion': 'West Brom',
  'Wolverhampton Wanderers': 'Wolves',
}

export function normalizeTeamName(team) {
  return teamAliases[team] ?? team
}

export function getDisplayTeamName(team) {
  const normalizedTeam = normalizeTeamName(team)
  return teamDisplayNames[normalizedTeam] ?? normalizedTeam
}

export function getClubLogoUrl(team) {
  const normalizedTeam = normalizeTeamName(team)
  const domain = teamDomains[normalizedTeam]
  if (!domain) return null
  return `https://www.google.com/s2/favicons?domain=${domain}&sz=128`
}

export function getTeamInitials(team) {
  return team
    .split(' ')
    .filter(Boolean)
    .map((part) => part[0]?.toUpperCase())
    .slice(0, 2)
    .join('')
}
