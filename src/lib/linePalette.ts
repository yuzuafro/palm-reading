export const PALM_LINE_IDS = ['life', 'head', 'heart'] as const

export type PalmLineId = (typeof PALM_LINE_IDS)[number]

export interface PalmLinePalette {
  color: string
  glow: string
  soft: string
  text: string
}

export const LINE_PALETTE: Record<PalmLineId, PalmLinePalette> = {
  life: {
    color: '#fb923c',
    glow: 'rgba(251, 146, 60, 0.45)',
    soft: 'rgba(251, 146, 60, 0.16)',
    text: '#fed7aa',
  },
  head: {
    color: '#60a5fa',
    glow: 'rgba(96, 165, 250, 0.45)',
    soft: 'rgba(96, 165, 250, 0.16)',
    text: '#bfdbfe',
  },
  heart: {
    color: '#f472b6',
    glow: 'rgba(244, 114, 182, 0.45)',
    soft: 'rgba(244, 114, 182, 0.16)',
    text: '#fbcfe8',
  },
}
