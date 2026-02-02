const clamp = (value, min, max) => Math.max(min, Math.min(max, value))

const hashSeed = (input) => {
  let hash = 0
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash << 5) - hash + input.charCodeAt(i)
    hash |= 0
  }
  return Math.abs(hash)
}

const wobble = (team, week, mod, offset) => {
  const seed = hashSeed(team) + week * 17 + offset
  return (seed % mod) - Math.floor(mod / 2)
}

export const teamList = [
  { team: 'Arsenal', color: '#ef233c', totalPoints: 86, predictedTotal: 83 },
  { team: 'Aston Villa', color: '#7d3c98', totalPoints: 63, predictedTotal: 60 },
  { team: 'Bournemouth', color: '#121212', totalPoints: 48, predictedTotal: 45 },
  { team: 'Brentford', color: '#ff3b3b', totalPoints: 52, predictedTotal: 51 },
  { team: 'Brighton', color: '#1f7aef', totalPoints: 58, predictedTotal: 55 },
  { team: 'Burnley', color: '#8d4b39', totalPoints: 42, predictedTotal: 40 },
  { team: 'Chelsea', color: '#034efc', totalPoints: 70, predictedTotal: 68 },
  { team: 'Crystal Palace', color: '#0033a0', totalPoints: 50, predictedTotal: 49 },
  { team: 'Everton', color: '#274690', totalPoints: 46, predictedTotal: 44 },
  { team: 'Fulham', color: '#1b1b1b', totalPoints: 49, predictedTotal: 47 },
  { team: 'Liverpool', color: '#c1121f', totalPoints: 88, predictedTotal: 85 },
  { team: 'Luton Town', color: '#f08a24', totalPoints: 38, predictedTotal: 36 },
  { team: 'Manchester City', color: '#6cabdd', totalPoints: 90, predictedTotal: 89 },
  { team: 'Manchester United', color: '#da291c', totalPoints: 66, predictedTotal: 64 },
  { team: 'Newcastle United', color: '#3a3a3a', totalPoints: 65, predictedTotal: 62 },
  { team: 'Nottingham Forest', color: '#e63946', totalPoints: 44, predictedTotal: 43 },
  { team: 'Sheffield United', color: '#c41e3a', totalPoints: 34, predictedTotal: 33 },
  { team: 'Tottenham', color: '#ffffff', totalPoints: 72, predictedTotal: 70 },
  { team: 'West Ham', color: '#7a263a', totalPoints: 56, predictedTotal: 53 },
  { team: 'Wolves', color: '#fcbf49', totalPoints: 47, predictedTotal: 46 }
]

export function buildStandings(gameweek) {
  const progress = clamp(gameweek / 38, 0, 1)

  return teamList.map((team) => {
    const basePoints = team.totalPoints * progress
    const basePredicted = team.predictedTotal * progress

    const points = Math.round(
      clamp(basePoints + wobble(team.team, gameweek, 7, 11), 0, team.totalPoints)
    )
    const predictedPoints = Math.round(
      clamp(basePredicted + wobble(team.team, gameweek, 9, 31), 0, team.predictedTotal)
    )

    return {
      team: team.team,
      color: team.color,
      points,
      predictedPoints
    }
  })
}
