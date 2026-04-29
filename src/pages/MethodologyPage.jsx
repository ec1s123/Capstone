import {
  Activity,
  BarChart3,
  BrainCircuit,
  FileOutput,
  Home,
  ShieldCheck,
  Target,
  TrendingUp,
} from 'lucide-react'

import { Badge } from '../components/ui/badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table'

const methodology = [
  {
    title: 'Historical Performance',
    description: 'Long-term team patterns, matchup history, and season-level trends shape the baseline view.',
    icon: Activity,
  },
  {
    title: 'Recent Form Tracking',
    description: 'Short-term momentum, consistency, and changes in output are factored into every fixture forecast.',
    icon: TrendingUp,
  },
  {
    title: 'Match Context',
    description: 'Home and away strength, scoring profiles, and defensive stability refine the prediction for each game.',
    icon: Home,
  },
  {
    title: 'Model-Driven Probabilities',
    description: 'Structured model outputs turn raw football data into clearer, more actionable prediction signals.',
    icon: BrainCircuit,
  },
]

const pipelineRows = [
  {
    step: '1',
    stage: 'Load and clean fixtures',
    whatHappens: 'Historical Premier League CSV files are read, normalized, date-parsed, and deduplicated by fixture.',
    userValue: 'The model starts from one consistent fixture history instead of mixed team names or duplicate matches.',
  },
  {
    step: '2',
    stage: 'Build pre-match features',
    whatHappens: 'Team identity, month, day of week, Bet365 odds, implied probabilities, market margin, and favorite strength are derived.',
    userValue: 'Predictions are based on information available before kickoff, not post-match stats.',
  },
  {
    step: '3',
    stage: 'Backtest latest season',
    whatHappens: 'Older seasons train the model, while the latest season is held out to compare model picks against actual results.',
    userValue: 'The dashboard can compare model behavior with the betting market on unseen fixtures.',
  },
  {
    step: '4',
    stage: 'Train softmax classifier',
    whatHappens: 'A three-class softmax model estimates home-win, draw, and away-win probabilities with L2 regularization.',
    userValue: 'Every fixture gets a full probability distribution rather than a single hard guess.',
  },
  {
    step: '5',
    stage: 'Generate reports',
    whatHappens: 'The model writes fixture-level probabilities, market probabilities, predictions, and model artifacts into app data files.',
    userValue: 'Tables, match cards, confidence badges, and detail panels all use the same prediction source.',
  },
]

const featureGroups = [
  {
    title: 'Team Identity',
    examples: 'Home team, away team',
    reason: 'Captures persistent strength, weakness, and home/away patterns for each club.',
  },
  {
    title: 'Calendar Context',
    examples: 'Month, day of week',
    reason: 'Adds light season timing context without relying on future match information.',
  },
  {
    title: 'Market Signal',
    examples: 'Bet365 home/draw/away odds',
    reason: 'Provides a strong pre-match benchmark that the model can learn with and compare against.',
  },
  {
    title: 'Derived Odds Features',
    examples: 'Implied probabilities, market margin, probability gap, favorite probability, log odds',
    reason: 'Turns raw odds into normalized signals that are easier for the model to use consistently.',
  },
]

const modelDetails = [
  ['Prediction target', 'Three outcomes: home win, draw, away win'],
  ['Model type', 'Custom softmax classifier trained in NumPy'],
  ['Optimization', 'Mini-batch Adam-style gradient updates'],
  ['Regularization', 'L2 penalty to reduce overfitting'],
  ['Training split', 'Historical seasons train; latest season backtests'],
  ['Production model', 'Retrained on all available played matches for the app output'],
]

const outputRows = [
  {
    output: 'Model probabilities',
    description: 'Home, draw, and away probabilities for every fixture.',
    usedBy: 'Confidence badges, probability charts, predicted table',
  },
  {
    output: 'Market probabilities',
    description: 'Bookmaker odds converted into normalized implied probabilities.',
    usedBy: 'Model-vs-market comparison, edge charts',
  },
  {
    output: 'Prediction correctness',
    description: 'Compares the model pick with the recorded full-time result.',
    usedBy: 'Accuracy cards, match rows, club breakdown',
  },
  {
    output: 'Fixture stats',
    description: 'Goals, shots, cards, corners, fouls, and halftime details when available.',
    usedBy: 'Post-match analysis panels only',
  },
]

export function MethodologyPage() {
  return (
    <section className="space-y-5">
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">How It Works</p>
        <h2 className="text-3xl font-semibold tracking-tight text-slate-900">
          How the prediction model turns fixtures into probabilities.
        </h2>
        <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
          The model is a pre-match softmax classifier. It learns from historical fixtures, team identity, calendar
          context, and bookmaker odds, then produces home-win, draw, and away-win probabilities for each match.
        </p>
      </div>

      <Card className="border-slate-200 bg-white shadow-sm">
        <CardContent className="grid gap-4 p-6 md:grid-cols-2 xl:grid-cols-4">
          {methodology.map((item) => {
            const Icon = item.icon
            return (
              <div key={item.title} className="rounded-xl border border-slate-200 bg-slate-50 p-5">
                <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-sm ring-1 ring-slate-200">
                  <Icon className="h-5 w-5 text-slate-700" />
                </div>
                <h3 className="text-base font-semibold text-slate-900">{item.title}</h3>
                <p className="mt-2 text-sm leading-6 text-slate-600">{item.description}</p>
              </div>
            )
          })}
        </CardContent>
      </Card>

      <Card className="border-slate-200 bg-white shadow-sm">
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle className="text-xl">Model Pipeline</CardTitle>
              <CardDescription className="mt-1">
                The app follows the same sequence each time predictions are regenerated from <code>src/MLMODEL.py</code>.
              </CardDescription>
            </div>
            <Badge variant="outline" className="gap-1 border-slate-200 bg-slate-50 text-slate-700">
              <BrainCircuit className="h-3.5 w-3.5" />
              Softmax
            </Badge>
          </div>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="overflow-hidden rounded-lg border border-slate-200">
            <Table>
              <TableHeader className="bg-slate-50">
                <TableRow className="border-b-0 hover:bg-transparent">
                  <TableHead className="w-[64px]">Step</TableHead>
                  <TableHead className="w-[220px]">Stage</TableHead>
                  <TableHead>What Happens</TableHead>
                  <TableHead>Why It Matters</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {pipelineRows.map((row) => (
                  <TableRow key={row.step}>
                    <TableCell className="font-semibold tabular-nums text-slate-900">{row.step}</TableCell>
                    <TableCell className="font-semibold text-slate-900">{row.stage}</TableCell>
                    <TableCell className="text-slate-700">{row.whatHappens}</TableCell>
                    <TableCell className="text-slate-600">{row.userValue}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <section className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <Card className="border-slate-200 bg-white shadow-sm">
          <CardHeader>
            <CardTitle className="text-xl">Feature Inputs</CardTitle>
            <CardDescription>
              Inputs are intentionally limited to fields known before the match starts.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3 pt-0 md:grid-cols-2">
            {featureGroups.map((group) => (
              <div key={group.title} className="rounded-lg border border-slate-200 bg-slate-50/70 p-4">
                <p className="text-sm font-semibold text-slate-900">{group.title}</p>
                <p className="mt-2 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Examples</p>
                <p className="mt-1 text-sm text-slate-700">{group.examples}</p>
                <p className="mt-3 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Purpose</p>
                <p className="mt-1 text-sm leading-6 text-slate-600">{group.reason}</p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="border-slate-200 bg-white shadow-sm">
          <CardHeader>
            <CardTitle className="text-xl">Training Setup</CardTitle>
            <CardDescription>Key implementation choices behind the model run.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3 pt-0">
            {modelDetails.map(([label, value]) => (
              <div key={label} className="flex items-start justify-between gap-4 border-b border-slate-200 pb-3 last:border-b-0 last:pb-0">
                <p className="text-sm font-semibold text-slate-900">{label}</p>
                <p className="max-w-[16rem] text-right text-sm text-slate-600">{value}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      </section>

      <Card className="border-slate-200 bg-white shadow-sm">
        <CardHeader>
          <CardTitle className="text-xl">How to Read the Output</CardTitle>
          <CardDescription>
            The model output is used differently depending on whether the screen is forecasting, comparing, or explaining a match.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 pt-0 md:grid-cols-2 xl:grid-cols-4">
          {outputRows.map((row, index) => {
            const icons = [Target, BarChart3, ShieldCheck, FileOutput]
            const Icon = icons[index]
            return (
              <div key={row.output} className="rounded-lg border border-slate-200 bg-slate-50/70 p-4">
                <div className="mb-3 flex h-9 w-9 items-center justify-center rounded-md border border-slate-200 bg-white text-slate-700">
                  <Icon className="h-4 w-4" />
                </div>
                <p className="text-sm font-semibold text-slate-900">{row.output}</p>
                <p className="mt-2 text-sm leading-6 text-slate-600">{row.description}</p>
                <p className="mt-3 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Used by</p>
                <p className="mt-1 text-sm text-slate-700">{row.usedBy}</p>
              </div>
            )
          })}
        </CardContent>
      </Card>

      <Card className="border-indigo-200 bg-indigo-50/50 shadow-sm">
        <CardContent className="grid gap-4 p-5 md:grid-cols-[auto_1fr]">
          <div className="flex h-11 w-11 items-center justify-center rounded-full border border-indigo-200 bg-white text-indigo-700">
            <ShieldCheck className="h-5 w-5" />
          </div>
          <div className="space-y-2">
            <p className="text-base font-semibold text-indigo-950">Leakage guardrail</p>
            <p className="max-w-4xl text-sm leading-6 text-indigo-900/85">
              Post-match fields such as goals, shots, corners, cards, fouls, and halftime scores are not used as model
              inputs. They appear only after prediction generation so users can analyze why a fixture played out the
              way it did.
            </p>
          </div>
        </CardContent>
      </Card>
    </section>
  )
}
