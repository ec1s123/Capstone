SELECT
  m.match_date,
  th.name AS home_team,
  ta.name AS away_team,
  m.home_goals,
  m.away_goals
FROM matches m
JOIN teams th ON m.home_team_id = th.team_id
JOIN teams ta ON m.away_team_id = ta.team_id
WHERE th.name = 'Arsenal'
   OR ta.name = 'Arsenal'
ORDER BY m.match_date DESC
LIMIT 10;