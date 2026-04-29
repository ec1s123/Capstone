import { cn } from '../../lib/utils'
import { formResultClass } from '../../lib/formatters'

export function FormChips({ results }) {
  const safeResults = Array.isArray(results) ? results : [null, null, null, null, null]

  return (
    <div className="flex items-center gap-1 whitespace-nowrap">
      {safeResults.map((result, index) => (
        <span
          key={`${result ?? 'na'}-${index}`}
          className={cn(
            'inline-flex h-5 w-5 items-center justify-center rounded-sm text-[9px] font-bold tracking-[0.04em]',
            formResultClass(result)
          )}
          aria-label={result ? `Result ${result}` : 'No result'}
          title={result ?? 'No result'}
        >
          {result ?? '-'}
        </span>
      ))}
    </div>
  )
}
