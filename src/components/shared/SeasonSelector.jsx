import { cn } from '../../lib/utils'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../ui/select'

export function SeasonSelector({ season, seasonOptions, onSeasonChange, className }) {
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
