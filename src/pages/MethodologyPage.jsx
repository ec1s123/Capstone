import { Activity, BrainCircuit, Home, TrendingUp } from 'lucide-react'

import { Card, CardContent } from '../components/ui/card'

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

export function MethodologyPage() {
  return (
    <section className="space-y-4">
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-[0.2em] text-slate-500">How It Works</p>
        <h2 className="text-3xl font-semibold tracking-tight text-slate-900">
          Every prediction is built on a structured approach.
        </h2>
        <p className="max-w-3xl text-sm text-muted-foreground md:text-base">
          Rather than relying on guesswork, the platform focuses on measurable signals that shape results across a
          full Premier League season.
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
    </section>
  )
}
