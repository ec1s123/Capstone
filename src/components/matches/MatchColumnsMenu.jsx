import { useEffect, useState } from 'react'
import { Columns3 } from 'lucide-react'

import { MATCH_COLUMN_DEFINITIONS } from '../../constants/matchColumns'

export function MatchColumnsMenu({ columnVisibility, onToggleColumn, onResetColumns }) {
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    if (!isOpen) return undefined
    const handlePointerDown = (event) => {
      const target = event.target
      if (!(target instanceof HTMLElement)) return
      if (!target.closest('[data-columns-menu-root]')) {
        setIsOpen(false)
      }
    }
    window.addEventListener('pointerdown', handlePointerDown)
    return () => window.removeEventListener('pointerdown', handlePointerDown)
  }, [isOpen])

  return (
    <div className="relative" data-columns-menu-root>
      <button
        type="button"
        className="inline-flex h-10 items-center gap-2 rounded-md border border-slate-300 bg-white px-3 text-sm font-semibold text-slate-700 hover:border-slate-400"
        onClick={() => setIsOpen((current) => !current)}
      >
        <Columns3 className="h-4 w-4" />
        Columns
      </button>
      {isOpen && (
        <div className="absolute right-0 z-20 mt-2 w-64 rounded-lg border border-slate-200 bg-white p-3 shadow-xl">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">Visible Columns</p>
            <button
              type="button"
              className="text-xs font-semibold text-slate-600 underline-offset-2 hover:underline"
              onClick={onResetColumns}
            >
              Reset
            </button>
          </div>
          <div className="grid grid-cols-1 gap-1.5">
            {MATCH_COLUMN_DEFINITIONS.map((column) => (
              <label
                key={`match-column-${column.key}`}
                className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
              >
                <input
                  type="checkbox"
                  checked={columnVisibility[column.key] !== false}
                  onChange={() => onToggleColumn(column.key)}
                />
                <span>{column.label}</span>
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
