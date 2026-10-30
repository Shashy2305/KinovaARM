/** The colour name the cameras give an object (object_colour.py) -> a screen colour, so the maps draw a white cup white and a blue cup blue. */
export const COLOUR_HEX: Record<string, string> = {
  white: '#f4f4f2', black: '#2a2a2e', grey: '#9aa0a6', red: '#e5484d', orange: '#f5a83a', yellow: '#f2d544',
  green: '#46c26b', blue: '#2f7be0', purple: '#9b6cf0', pink: '#f08cc0', brown: '#8a5a3c',
}

export function colourHex(name?: string | null): string | null {
  return name && COLOUR_HEX[name] ? COLOUR_HEX[name] : null
}

/** "white cup" when the colour is known, else "cup". */
export function objectName(label: string, colour?: string | null): string {
  return colour && colour !== 'unknown' ? `${colour} ${label}` : label
}
