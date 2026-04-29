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

export function normalizeTeamName(team) {
  return teamAliases[team] ?? team
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
