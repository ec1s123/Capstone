import { useEffect, useMemo, useState } from 'react'

import { SeasonSelector } from '../components/shared/SeasonSelector'
import { FinalModelTableCard } from '../components/standings/FinalModelTableCard'
import { MatchDetailsDrawer } from '../components/matches/MatchDetailsDrawer'
import {
  GameTexturePanel,
  MatchSummaryTiles,
  ModelEdgePanel,
  OutcomeMixChart,
  TeamSignalPanel,
} from '../components/matches/MatchInsightsPanels'
import { buildMatchPageInsightData } from '../lib/matchInsights'
import { cn } from '../lib/utils'

export function ModelOutputPage({ favoriteTeam, matches, modelOutputTable, season, seasonOptions, onSeasonChange }) {
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
