import { useState } from 'react'

import { cn } from '../../lib/utils'
import { getClubLogoUrl, getTeamInitials, normalizeTeamName } from '../../lib/teamUtils'

const logoSizeMap = {
  sm: {
    fallback: 'h-5 w-5 text-[8px]',
    image: 'h-5 w-5',
  },
  md: {
    fallback: 'h-6 w-6 text-[9px]',
    image: 'h-6 w-6',
  },
  lg: {
    fallback: 'h-10 w-10 text-xs',
    image: 'h-10 w-10',
  },
  xl: {
    fallback: 'h-14 w-14 text-sm',
    image: 'h-14 w-14',
  },
}

export function ClubLogo({ team, size = 'md' }) {
  const [hasError, setHasError] = useState(false)
  const normalizedTeam = normalizeTeamName(team)
  const logoUrl = getClubLogoUrl(normalizedTeam)
  const initials = getTeamInitials(normalizedTeam)
  const sizeClass = logoSizeMap[size] ?? logoSizeMap.md

  if (!logoUrl || hasError) {
    return (
      <span
        className={cn(
          'inline-flex shrink-0 items-center justify-center rounded-full border border-slate-300 bg-white font-semibold text-slate-700',
          sizeClass.fallback
        )}
      >
        {initials}
      </span>
    )
  }

  return (
    <img
      src={logoUrl}
      alt={`${normalizedTeam} crest`}
      className={cn(
        'shrink-0 rounded-full border border-slate-200 bg-white object-cover p-[1px]',
        sizeClass.image
      )}
      loading="lazy"
      onError={() => setHasError(true)}
    />
  )
}
