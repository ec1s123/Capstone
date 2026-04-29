import { NavLink } from 'react-router-dom'

import { navItems } from '../../constants/navigation'
import { cn } from '../../lib/utils'

export function AppNavigation() {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/90 backdrop-blur-sm">
      <div className="mx-auto w-full max-w-[1320px] px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-3 py-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Premier League Predictor</p>
            <p className="text-lg font-semibold tracking-tight text-slate-900">Data-driven match insights</p>
          </div>
          <nav className="flex gap-2 overflow-x-auto pb-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  cn(
                    'rounded-full border px-3 py-1.5 text-xs font-semibold uppercase tracking-[0.12em] transition-colors',
                    isActive
                      ? 'border-slate-900 bg-slate-900 text-white'
                      : 'border-slate-300 bg-white text-slate-700 hover:border-slate-400 hover:text-slate-900'
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </div>
    </header>
  )
}
