import { Badge } from '../components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import premOddsMetrics from '../data/prem_odds_metrics.json'
import { formatPercent, formatSigned } from '../lib/formatters'
import { barPercent, probabilityBarHeight } from '../lib/matchInsights'
import { cn } from '../lib/utils'

const methodologyPipeline = [
  {
    step: '01',
    title: 'Load Historical Fixtures',
    description: 'CSV files are normalised into one fixture set with consistent teams, dates, odds, results, and seasons.',
  },
  {
    step: '02',
    title: 'Build Pre-Match Features',
    description: 'The model only sees team identity, date context, and 1X2 market odds that would be known before kick-off.',
  },
  {
    step: '03',
    title: 'Backtest Latest Season',
    description: 'Older seasons train the model; the latest available season is held out to test accuracy and probability quality.',
  },
  {
    step: '04',
    title: 'Publish Probabilities',
    description: 'The production model is refit on every played fixture and outputs home, draw, and away probabilities.',
  },
]

const methodologyFeatureGroups = [
  {
    title: 'Team Identity',
    description: 'Home and away teams are encoded separately so the model can learn long-run club strength and venue effects.',
  },
  {
    title: 'Calendar Context',
    description: 'Month and day-of-week signals let the model capture seasonal scheduling patterns without using match outcomes.',
  },
  {
    title: 'Market Odds',
    description: 'Bet365 home, draw, and away odds are transformed into probability and log-odds features for pre-match context.',
  },
  {
    title: 'Excluded Match Stats',
    description: 'Goals, shots, cards, corners, fouls, and half-time scores are kept out of training to avoid target leakage.',
  },
]

function MethodologyStatCard({ label, value, detail }) {
  return (
    <div className="bg-white/70 p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-semibold tracking-tight text-slate-900">{value}</p>
      <p className="mt-1 text-sm text-slate-600">{detail}</p>
    </div>
  )
}

function MethodologyModelComparison({ metrics }) {
  const models = [
    {
      key: 'teams_only_softmax',
      label: 'Teams Only',
      description: 'Baseline using club identity and date context.',
      tone: 'bg-slate-500',
    },
    {
      key: 'teams_plus_odds_softmax',
      label: 'Teams + Odds',
      description: 'Production-style model with pre-match odds features.',
      tone: 'bg-sky-500',
    },
    {
      key: 'market_implied_probabilities',
      label: 'Market Implied',
      description: 'Bookmaker probability benchmark after overround normalisation.',
      tone: 'bg-amber-500',
    },
  ].map((model) => ({ ...model, ...metrics.models[model.key] }))
  const bestAccuracy = Math.max(...models.map((model) => model.accuracy))
  const bestLogLoss = Math.min(...models.map((model) => model.log_loss))

  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Backtest Comparison</CardTitle>
        <CardDescription>
          Holdout performance on the {metrics.test_season} season. Accuracy rewards correct picks; log loss rewards calibrated probabilities.
        </CardDescription>
      </CardHeader>
      <CardContent className="hairline-list">
        {models.map((model) => (
          <div key={model.key} className="p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="font-semibold text-slate-900">{model.label}</p>
                <p className="mt-1 text-sm text-slate-600">{model.description}</p>
              </div>
              <div className="flex flex-wrap gap-2">
                {model.accuracy === bestAccuracy && (
                  <Badge variant="outline" className="border-emerald-200 bg-emerald-50 text-emerald-700">
                    Best accuracy
                  </Badge>
                )}
                {model.log_loss === bestLogLoss && (
                  <Badge variant="outline" className="border-sky-200 bg-sky-50 text-sky-700">
                    Best calibration
                  </Badge>
                )}
              </div>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-2">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-600">Accuracy</span>
                  <span className="font-semibold tabular-nums text-slate-900">{formatPercent(model.accuracy)}</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-white">
                  <div className={cn('h-full rounded-full', model.tone)} style={{ width: probabilityBarHeight(model.accuracy) }} />
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-600">Log Loss</span>
                  <span className="font-semibold tabular-nums text-slate-900">{model.log_loss.toFixed(3)}</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-white">
                  <div
                    className={cn('h-full rounded-full', model.tone)}
                    style={{ width: `${Math.max((bestLogLoss / model.log_loss) * 100, 4)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

function MethodologyPipeline() {
  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Pipeline</CardTitle>
        <CardDescription>How raw football-data CSVs become probabilities in the app.</CardDescription>
      </CardHeader>
      <CardContent className="hairline-grid md-cols-2 xl-cols-4 grid gap-0 p-0 md:grid-cols-2 xl:grid-cols-4">
        {methodologyPipeline.map((item) => (
          <div key={item.step} className="bg-white/60 p-5">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">{item.step}</p>
            <h3 className="mt-3 text-base font-semibold text-slate-900">{item.title}</h3>
            <p className="mt-2 text-sm leading-6 text-slate-600">{item.description}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

function MethodologyFeatureGrid() {
  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Feature Design</CardTitle>
        <CardDescription>What is allowed into the model, and what is deliberately kept out.</CardDescription>
      </CardHeader>
      <CardContent className="hairline-grid md-cols-2 xl-cols-4 grid gap-0 p-0 md:grid-cols-2 xl:grid-cols-4">
        {methodologyFeatureGroups.map((item) => (
          <div key={item.title} className="bg-white/60 p-5">
            <h3 className="text-base font-semibold text-slate-900">{item.title}</h3>
            <p className="mt-2 text-sm leading-6 text-slate-600">{item.description}</p>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}

function MethodologySeasonCoverage({ counts }) {
  const entries = Object.entries(counts)
  const maxMatches = Math.max(...entries.map(([, count]) => count), 1)

  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">Season Coverage</CardTitle>
        <CardDescription>Historical match volume used to train and backtest the probability model.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="hairline-grid md-cols-2 lg-cols-3 xl-cols-4 grid gap-0 overflow-hidden rounded-lg bg-white/70 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {entries.map(([season, count]) => (
            <div key={season} className="grid grid-cols-[4.5rem_1fr_3rem] items-center gap-2 p-3 text-sm">
              <span className="font-medium text-slate-700">{season}</span>
              <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                <div className="h-full rounded-full bg-slate-900" style={{ width: barPercent(count, maxMatches) }} />
              </div>
              <span className="text-right tabular-nums text-slate-600">{count}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function MethodologyPage() {
  const oddsModel = premOddsMetrics.models.teams_plus_odds_softmax
  const marketModel = premOddsMetrics.models.market_implied_probabilities
  const accuracyGap = oddsModel.accuracy - premOddsMetrics.models.teams_only_softmax.accuracy

  return (
    <section className="space-y-4">
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">How It Works</p>
        <h2 className="text-3xl font-semibold tracking-tight text-slate-900">
          From historical fixtures to calibrated match probabilities.
        </h2>
        <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
          The model is intentionally built around information available before kick-off, then tested against a held-out
          season so users can judge how much signal it adds beyond team history and market odds.
        </p>
      </div>

      <section className="hairline-grid md-cols-2 xl-cols-4 grid gap-0 overflow-hidden rounded-lg bg-white/70 md:grid-cols-2 xl:grid-cols-4">
        <MethodologyStatCard
          label="Matches Used"
          value={premOddsMetrics.matches_used.toLocaleString()}
          detail={`${premOddsMetrics.seasons_used} seasons from ${premOddsMetrics.data_start} to ${premOddsMetrics.data_end}`}
        />
        <MethodologyStatCard
          label="Training Split"
          value={premOddsMetrics.train_matches.toLocaleString()}
          detail={`${premOddsMetrics.test_matches} fixtures held out for ${premOddsMetrics.test_season}`}
        />
        <MethodologyStatCard
          label="Production Accuracy"
          value={formatPercent(oddsModel.accuracy)}
          detail={`${formatSigned(accuracyGap * 100, 1)} pts versus teams-only baseline`}
        />
        <MethodologyStatCard
          label="Market Benchmark"
          value={formatPercent(marketModel.accuracy)}
          detail={`Market log loss ${marketModel.log_loss.toFixed(3)} on the same holdout season`}
        />
      </section>

      <MethodologyPipeline />
      <MethodologyModelComparison metrics={premOddsMetrics} />

      <div className="grid gap-4 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <MethodologyFeatureGrid />
        <Card className="border-slate-200 bg-white shadow-sm">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg">Leakage Controls</CardTitle>
            <CardDescription>Why the model avoids using data that would only be known after the final whistle.</CardDescription>
          </CardHeader>
          <CardContent className="hairline-list">
            {premOddsMetrics.notes.map((note) => (
              <div key={note} className="p-4">
                <p className="text-sm leading-6 text-slate-700">{note}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <MethodologySeasonCoverage counts={premOddsMetrics.season_match_counts} />
    </section>
  )
}
