import { Badge } from '../components/ui/badge'
import { SeasonSelector } from '../components/shared/SeasonSelector'
import { FavoriteTeamCard } from '../components/standings/FavoriteTeamCard'
import { PerformerList } from '../components/standings/PerformerList'

export function OverviewPage({
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
