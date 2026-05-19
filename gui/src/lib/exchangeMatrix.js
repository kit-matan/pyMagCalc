/**
 * Pure helpers that turn an interaction record + parameter map into a 3×3
 * exchange matrix for display. Two flavors:
 *   - numeric: evaluates parameter references to floats.
 *   - symbolic: keeps parameter names as strings, with cosmetic cleanups so
 *     the rendered matrix looks like algebra rather than backend output.
 *
 * Extracted from App.jsx so the same logic can be reused by the bond popover
 * and the interactions table without dragging in App-level state.
 */

export function calculateExchangeMatrix(inter, numericParams = {}) {
  const type = inter.type
  const value = inter.value

  const evalParam = (val) => {
    if (typeof val === 'number') return val
    if (!val) return 0.0
    if (numericParams[val] !== undefined) return numericParams[val]
    const f = parseFloat(val)
    if (!isNaN(f)) return f
    return 0.0
  }

  try {
    if (type === 'heisenberg') {
      const j = evalParam(value)
      return [[j, 0, 0], [0, j, 0], [0, 0, j]]
    }
    if (type === 'dm' || type === 'dm_interaction') {
      let vec = [0, 0, 0]
      if (Array.isArray(value)) vec = value.map(evalParam)
      else if (typeof value === 'string') vec = value.split(',').map((s) => evalParam(s.trim()))
      const [dx, dy, dz] = vec
      return [[0, dz, -dy], [-dz, 0, dx], [dy, -dx, 0]]
    }
    if (type === 'anisotropic' || type === 'anisotropic_exchange') {
      let vec = [0, 0, 0]
      if (Array.isArray(value)) vec = value.map(evalParam)
      else if (typeof value === 'string') vec = value.split(',').map((s) => evalParam(s.trim()))
      const [jx, jy, jz] = vec
      return [[jx, 0, 0], [0, jy, 0], [0, 0, jz]]
    }
  } catch (e) {
    console.error('Matrix Calc Error', e)
    return null
  }
  return null
}

export function calculateExchangeMatrixSymbolic(inter, numericParams = {}) {
  const type = inter.type
  const value = inter.value

  const getSymbol = (val) => {
    if (val === undefined || val === null) return 0
    if (typeof val === 'number') {
      if (Math.abs(val) < 1e-10) return 0
      return val
    }

    let s = String(val).trim()

    // Tiny scientific-notation values → 0
    const smallFloatRegex = /[+-]?\d+\.?\d*[eE]-[1-9]\d+/g
    s = s.replace(smallFloatRegex, (match) => (Math.abs(parseFloat(match)) < 1e-10 ? '0' : match))

    // Round any decimals to 5 places; normalize leading "." / "-."
    const floatRegex = /[+-]?\d*\.\d+(?:[eE][+-]?\d+)?/g
    s = s.replace(floatRegex, (match) => {
      const f = parseFloat(match)
      if (Math.abs(f) < 1e-10) return '0'
      const rounded = Number(f.toFixed(5))
      let str = String(rounded)
      if (str.startsWith('.')) return '0' + str
      if (str.startsWith('-.')) return '-0' + str.substring(1)
      return str
    })

    // Cosmetic algebraic simplifications for display
    s = s.replace(/\b0\s*\*\s*[a-zA-Z0-9_]+/g, '0')
    s = s.replace(/[a-zA-Z0-9_]+\s*\*\s*0\b/g, '0')
    s = s.replace(/\+\s*0\b/g, '')
    s = s.replace(/\b0\s*\+\s*/g, '')
    s = s.replace(/\-\s*0\b/g, '')
    s = s.replace(/\b0\s*\-\s*/g, '-')

    if (s.trim() === '' || s === '-0' || s === '-0.0') s = '0'

    if (s.startsWith('1.0*')) s = s.substring(4)
    else if (s.startsWith('1*')) s = s.substring(2)

    if (s.startsWith('-1.0*')) s = '-' + s.substring(5)
    else if (s.startsWith('-1*')) s = '-' + s.substring(3)

    if (s === '0' || s === '0.0') return 0

    if (!isNaN(parseFloat(s)) && isFinite(s)) {
      if (Math.abs(parseFloat(s)) < 1e-10) return 0
      return s
    }
    return s
  }

  const neg = (v) => {
    if (v === 0) return 0
    if (typeof v === 'string') {
      if (v.startsWith('-')) return v.substring(1)
      return `-${v}`
    }
    return -v
  }

  try {
    if (type === 'heisenberg') {
      const j = getSymbol(value)
      return [[j, 0, 0], [0, j, 0], [0, 0, j]]
    }
    if (type === 'dm' || type === 'dm_interaction') {
      let vec = [0, 0, 0]
      if (Array.isArray(value)) vec = value.map(getSymbol)
      else if (typeof value === 'string') vec = value.split(',').map(getSymbol)
      const [dx, dy, dz] = vec
      return [[0, dz, neg(dy)], [neg(dz), 0, dx], [dy, neg(dx), 0]]
    }
    if (type === 'anisotropic' || type === 'anisotropic_exchange') {
      let vec = [0, 0, 0]
      if (Array.isArray(value)) vec = value.map(getSymbol)
      else if (typeof value === 'string') vec = value.split(',').map(getSymbol)
      const [jx, jy, jz] = vec
      return [[jx, 0, 0], [0, jy, 0], [0, 0, jz]]
    }
  } catch (e) {
    console.error('Matrix Calc Error', e)
    return null
  }
  return null
}
