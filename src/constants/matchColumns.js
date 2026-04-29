export const MATCH_COLUMN_STORAGE_KEY = 'premier_predict.matches.columns.v1'

export const MATCH_COLUMN_DEFINITIONS = [
  { key: 'date', label: 'Date' },
  { key: 'home', label: 'Home' },
  { key: 'away', label: 'Away' },
  { key: 'score', label: 'Score' },
  { key: 'result', label: 'Result' },
  { key: 'modelPick', label: 'Model Pick' },
  { key: 'confidence', label: 'Confidence' },
  { key: 'prediction', label: 'Prediction' },
]

export function defaultMatchColumnVisibility() {
  return Object.fromEntries(MATCH_COLUMN_DEFINITIONS.map((column) => [column.key, true]))
}

export function sanitizeMatchColumnVisibility(rawVisibility) {
  const defaults = defaultMatchColumnVisibility()
  if (!rawVisibility || typeof rawVisibility !== 'object') return defaults

  return MATCH_COLUMN_DEFINITIONS.reduce((accumulator, column) => {
    const rawValue = rawVisibility[column.key]
    accumulator[column.key] = typeof rawValue === 'boolean' ? rawValue : defaults[column.key]
    return accumulator
  }, {})
}

export function readCachedMatchColumnVisibility() {
  if (typeof window === 'undefined') return defaultMatchColumnVisibility()
  try {
    const cachedValue = window.localStorage.getItem(MATCH_COLUMN_STORAGE_KEY)
    if (!cachedValue) return defaultMatchColumnVisibility()
    return sanitizeMatchColumnVisibility(JSON.parse(cachedValue))
  } catch {
    return defaultMatchColumnVisibility()
  }
}
