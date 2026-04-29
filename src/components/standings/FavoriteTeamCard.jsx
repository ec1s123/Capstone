import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select'
import { ClubLogo } from '../shared/ClubLogo'

export function FavoriteTeamCard({ favoriteTeam, onFavoriteTeamChange, favoriteSnapshot, teamOptions }) {
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
