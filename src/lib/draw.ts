import { HandLandmarker, type NormalizedLandmark } from '@mediapipe/tasks-vision'
import { LINE_PALETTE, type PalmLineId } from './linePalette'
import type { PalmLineInput, PalmReading, Point } from './palmistry'

interface PixelPoint {
  x: number
  y: number
}

interface PalmTextureData {
  image: HTMLCanvasElement
  ridgeMap: Uint8ClampedArray
  sourceX: number
  sourceY: number
  sourceWidth: number
  sourceHeight: number
  maskPoints: PixelPoint[]
  updatedAt: number
}

interface PalmBasis {
  wrist: PixelPoint
  thumbMcp: PixelPoint
  indexMcp: PixelPoint
  middleMcp: PixelPoint
  ringMcp: PixelPoint
  pinkyMcp: PixelPoint
  palmCenter: PixelPoint
  palmWidth: number
  palmHeight: number
  xAxis: PixelPoint
  yAxis: PixelPoint
  lineJunction: PixelPoint
  localPoint: (u: number, v: number) => PixelPoint
  toLocalPoint: (point: PixelPoint) => PixelPoint
}

export interface PalmFrameAnalysis {
  texture: PalmTextureData
  detectedLines: PalmLineInput[]
}

const DETAIL_SIZE = 280
const TEXTURE_REFRESH_MS = 45
const VALLEY_DIRECTIONS: PixelPoint[] = [
  { x: 1, y: 0 },
  { x: 0.866, y: 0.5 },
  { x: 0.5, y: 0.866 },
  { x: 0, y: 1 },
  { x: -0.5, y: 0.866 },
  { x: -0.866, y: 0.5 },
]
const analysisCanvas = document.createElement('canvas')
const textureCanvas = document.createElement('canvas')

let cachedAnalysis:
  | {
      anchorKey: string
      result: PalmFrameAnalysis
    }
  | null = null

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

const distance = (from: PixelPoint, to: PixelPoint) => Math.hypot(to.x - from.x, to.y - from.y)

const averagePoint = (points: PixelPoint[]): PixelPoint => ({
  x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
  y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
})

const normalizeVector = (point: PixelPoint): PixelPoint => {
  const length = Math.hypot(point.x, point.y) || 1
  return {
    x: point.x / length,
    y: point.y / length,
  }
}

const perpendicular = (vector: PixelPoint): PixelPoint => ({
  x: -vector.y,
  y: vector.x,
})

const add = (left: PixelPoint, right: PixelPoint): PixelPoint => ({
  x: left.x + right.x,
  y: left.y + right.y,
})

const subtract = (left: PixelPoint, right: PixelPoint): PixelPoint => ({
  x: left.x - right.x,
  y: left.y - right.y,
})

const scale = (point: PixelPoint, factor: number): PixelPoint => ({
  x: point.x * factor,
  y: point.y * factor,
})

const dot = (left: PixelPoint, right: PixelPoint) => left.x * right.x + left.y * right.y

const interpolatePoint = (from: PixelPoint, to: PixelPoint, amount: number): PixelPoint => ({
  x: from.x + (to.x - from.x) * amount,
  y: from.y + (to.y - from.y) * amount,
})

const offsetPoint = (point: PixelPoint, direction: PixelPoint, amount: number): PixelPoint =>
  add(point, scale(direction, amount))

const scalePolygon = (points: PixelPoint[], factor: number): PixelPoint[] => {
  const center = averagePoint(points)
  return points.map((point) => ({
    x: center.x + (point.x - center.x) * factor,
    y: center.y + (point.y - center.y) * factor,
  }))
}

const densifyPoints = (points: PixelPoint[], samples = 32): PixelPoint[] => {
  if (points.length < 2) {
    return points
  }

  const densePoints: PixelPoint[] = []
  const segments = points.length - 1

  for (let index = 0; index < samples; index += 1) {
    const progress = index / (samples - 1)
    const segmentIndex = Math.min(segments - 1, Math.floor(progress * segments))
    const localProgress = progress * segments - segmentIndex
    densePoints.push(
      interpolatePoint(points[segmentIndex], points[segmentIndex + 1], localProgress),
    )
  }

  return densePoints
}

const smoothPoints = (points: PixelPoint[]): PixelPoint[] =>
  points.map((point, index) => {
    if (index === 0 || index === points.length - 1) {
      return point
    }

    return {
      x: (points[index - 1].x + point.x * 2 + points[index + 1].x) / 4,
      y: (points[index - 1].y + point.y * 2 + points[index + 1].y) / 4,
    }
  })

const repeatSmoothPoints = (points: PixelPoint[], iterations: number): PixelPoint[] => {
  let current = points
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    current = smoothPoints(current)
  }

  return current
}

const chaikinSmoothPoints = (points: PixelPoint[], iterations = 2): PixelPoint[] => {
  let current = points

  for (let iteration = 0; iteration < iterations; iteration += 1) {
    if (current.length < 3) {
      return current
    }

    const next: PixelPoint[] = [current[0]]
    for (let index = 0; index < current.length - 1; index += 1) {
      const point = current[index]
      const following = current[index + 1]
      next.push(interpolatePoint(point, following, 0.25))
      next.push(interpolatePoint(point, following, 0.75))
    }
    next.push(current[current.length - 1])
    current = next
  }

  return current
}

const toPixelPoint = (
  landmark: Pick<NormalizedLandmark, 'x' | 'y'>,
  width: number,
  height: number,
): PixelPoint => ({
  x: landmark.x * width,
  y: landmark.y * height,
})

const toNormalizedPoint = (point: PixelPoint, width: number, height: number): Point => ({
  x: point.x / width,
  y: point.y / height,
})

const sampleScalar = (
  values: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
) => {
  const sampleX = Math.round(x)
  const sampleY = Math.round(y)

  if (sampleX < 0 || sampleY < 0 || sampleX >= width || sampleY >= height) {
    return 0
  }

  return values[sampleY * width + sampleX]
}

const createRoundedRect = (
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
) => {
  context.beginPath()
  context.moveTo(x + radius, y)
  context.lineTo(x + width - radius, y)
  context.quadraticCurveTo(x + width, y, x + width, y + radius)
  context.lineTo(x + width, y + height - radius)
  context.quadraticCurveTo(x + width, y + height, x + width - radius, y + height)
  context.lineTo(x + radius, y + height)
  context.quadraticCurveTo(x, y + height, x, y + height - radius)
  context.lineTo(x, y + radius)
  context.quadraticCurveTo(x, y, x + radius, y)
  context.closePath()
}

const createPalmMask = (landmarks: NormalizedLandmark[], width: number, height: number) =>
  scalePolygon(
    [landmarks[2], landmarks[5], landmarks[9], landmarks[13], landmarks[17], landmarks[0]].map(
      (landmark) => toPixelPoint(landmark, width, height),
    ),
    1.12,
  )

const createAnchorKey = (landmarks: NormalizedLandmark[]) =>
  [0, 2, 5, 9, 13, 17]
    .map((index) => `${landmarks[index].x.toFixed(4)}:${landmarks[index].y.toFixed(4)}`)
    .join('|')

const buildPalmTexture = (
  video: HTMLVideoElement,
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
): PalmTextureData | null => {
  const palmMask = createPalmMask(landmarks, width, height)
  const xValues = palmMask.map((point) => point.x)
  const yValues = palmMask.map((point) => point.y)
  const sourceX = clamp(Math.floor(Math.min(...xValues)), 0, width - 1)
  const sourceY = clamp(Math.floor(Math.min(...yValues)), 0, height - 1)
  const maxX = clamp(Math.ceil(Math.max(...xValues)), 1, width)
  const maxY = clamp(Math.ceil(Math.max(...yValues)), 1, height)
  const sourceWidth = Math.max(12, maxX - sourceX)
  const sourceHeight = Math.max(12, maxY - sourceY)

  analysisCanvas.width = DETAIL_SIZE
  analysisCanvas.height = DETAIL_SIZE
  textureCanvas.width = DETAIL_SIZE
  textureCanvas.height = DETAIL_SIZE

  const analysisContext = analysisCanvas.getContext('2d', { willReadFrequently: true })
  const textureContext = textureCanvas.getContext('2d')
  if (!analysisContext || !textureContext) {
    return null
  }

  analysisContext.clearRect(0, 0, DETAIL_SIZE, DETAIL_SIZE)
  analysisContext.drawImage(
    video,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    0,
    0,
    DETAIL_SIZE,
    DETAIL_SIZE,
  )
  const rawImage = analysisContext.getImageData(0, 0, DETAIL_SIZE, DETAIL_SIZE)
  analysisContext.clearRect(0, 0, DETAIL_SIZE, DETAIL_SIZE)
  analysisContext.filter = 'grayscale(1) contrast(1.9) brightness(1.06)'
  analysisContext.drawImage(
    video,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    0,
    0,
    DETAIL_SIZE,
    DETAIL_SIZE,
  )
  analysisContext.filter = 'none'

  const image = analysisContext.getImageData(0, 0, DETAIL_SIZE, DETAIL_SIZE)
  const texture = textureContext.createImageData(DETAIL_SIZE, DETAIL_SIZE)
  const ridgeMap = new Uint8ClampedArray(DETAIL_SIZE * DETAIL_SIZE)
  const luminance = new Float32Array(DETAIL_SIZE * DETAIL_SIZE)
  let averageWarmth = 0

  for (let index = 0; index < luminance.length; index += 1) {
    const pixelIndex = index * 4
    const red = image.data[pixelIndex]
    const green = image.data[pixelIndex + 1]
    const blue = image.data[pixelIndex + 2]
    luminance[index] = red * 0.299 + green * 0.587 + blue * 0.114

    const rawRed = rawImage.data[pixelIndex]
    const rawGreen = rawImage.data[pixelIndex + 1]
    const rawBlue = rawImage.data[pixelIndex + 2]
    averageWarmth += rawRed - (rawGreen + rawBlue) * 0.5
  }
  averageWarmth /= luminance.length

  for (let y = 2; y < DETAIL_SIZE - 2; y += 1) {
    for (let x = 2; x < DETAIL_SIZE - 2; x += 1) {
      const index = y * DETAIL_SIZE + x
      const center = luminance[index]
      let bestScore = 0

      for (const dir of VALLEY_DIRECTIONS) {
        const tang = perpendicular(dir)

        // Sample across the crease (perpendicular to crease direction)
        const left1 = sampleScalar(luminance, DETAIL_SIZE, DETAIL_SIZE, x + dir.x, y + dir.y)
        const right1 = sampleScalar(luminance, DETAIL_SIZE, DETAIL_SIZE, x - dir.x, y - dir.y)
        const n1 = (left1 + right1) * 0.5
        const left3 = sampleScalar(luminance, DETAIL_SIZE, DETAIL_SIZE, x + dir.x * 3, y + dir.y * 3)
        const right3 = sampleScalar(luminance, DETAIL_SIZE, DETAIL_SIZE, x - dir.x * 3, y - dir.y * 3)
        const n3 = (left3 + right3) * 0.5
        const left8 = sampleScalar(luminance, DETAIL_SIZE, DETAIL_SIZE, x + dir.x * 8, y + dir.y * 8)
        const right8 = sampleScalar(luminance, DETAIL_SIZE, DETAIL_SIZE, x - dir.x * 8, y - dir.y * 8)
        const n8 = (left8 + right8) * 0.5

        // Sample along the crease (tangent direction) — also dark for a true line
        const t3 =
          (sampleScalar(luminance, DETAIL_SIZE, DETAIL_SIZE, x + tang.x * 3, y + tang.y * 3) +
            sampleScalar(luminance, DETAIL_SIZE, DETAIL_SIZE, x - tang.x * 3, y - tang.y * 3)) *
          0.5

        // depth: how much darker center is than far background
        const depth = n8 - center
        if (depth < 6) {
          continue
        }

        // nearRec: brightness recovered at 1 px from center
        //   Crease  ≈ 0.7–1.0  (sharp wall, recovers immediately)
        //   Vessel  ≈ 0.1–0.4  (still inside the wide dark region)
        const nearRec = clamp((n1 - center) / depth, 0, 1)

        // midRec: brightness recovered at 3 px
        //   Crease  ≈ 0.9–1.0  (fully back to skin)
        //   Vessel  ≈ 0.4–0.65 (still noticeably dark)
        const midRec = clamp((n3 - center) / depth, 0, 1)

        // lineSup: along-direction is also dark → confirms linear structure, not a dot/blob
        const lineSup = clamp((n8 - t3) / Math.max(depth, 1), 0, 1)
        const symmetry = clamp(1 - Math.abs(left1 - right1) / Math.max(depth, 1), 0, 1)

        if (nearRec < 0.18 || midRec < 0.26 || lineSup < 0.12 || symmetry < 0.18) {
          continue
        }

        // Prefer narrow, symmetric creases. Wide cool-toned valleys tend to be veins.
        const score =
          depth *
          Math.pow(nearRec, 1.25) *
          (0.42 + midRec * 0.58) *
          (0.26 + lineSup * 0.74) *
          (0.32 + symmetry * 0.68)

        if (score > bestScore) {
          bestScore = score
        }
      }

      const pixelIndex = index * 4
      const rawRed = rawImage.data[pixelIndex]
      const rawGreen = rawImage.data[pixelIndex + 1]
      const rawBlue = rawImage.data[pixelIndex + 2]
      const warmth = rawRed - (rawGreen + rawBlue) * 0.5
      const coolPenalty = clamp((averageWarmth - warmth - 10) / 24, 0, 1)
      const correctedScore = bestScore * (1 - coolPenalty * 0.4)
      const alpha = clamp(Math.round((correctedScore - 8) * 4.2), 0, 190)

      if (alpha === 0) {
        continue
      }

      texture.data[pixelIndex] = 244
      texture.data[pixelIndex + 1] = 236
      texture.data[pixelIndex + 2] = 255
      texture.data[pixelIndex + 3] = alpha
      ridgeMap[index] = alpha
    }
  }

  textureContext.putImageData(texture, 0, 0)
  return {
    image: textureCanvas,
    ridgeMap,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    maskPoints: palmMask,
    updatedAt: performance.now(),
  }
}

const createPalmBasis = (
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
): PalmBasis => {
  const wrist = toPixelPoint(landmarks[0], width, height)
  const thumbMcp = toPixelPoint(landmarks[2], width, height)
  const indexMcp = toPixelPoint(landmarks[5], width, height)
  const middleMcp = toPixelPoint(landmarks[9], width, height)
  const ringMcp = toPixelPoint(landmarks[13], width, height)
  const pinkyMcp = toPixelPoint(landmarks[17], width, height)
  const palmTopCenter = averagePoint([indexMcp, middleMcp, ringMcp, pinkyMcp])
  const palmCenter = averagePoint([wrist, thumbMcp, indexMcp, middleMcp, ringMcp, pinkyMcp])
  const xAxis = normalizeVector(subtract(thumbMcp, pinkyMcp))
  const yAxis = normalizeVector(subtract(wrist, palmTopCenter))
  const palmWidth = distance(pinkyMcp, thumbMcp)
  const palmHeight = distance(palmTopCenter, wrist)

  const localPoint = (u: number, v: number): PixelPoint =>
    add(palmCenter, add(scale(xAxis, u * palmWidth), scale(yAxis, v * palmHeight)))

  const toLocalPoint = (point: PixelPoint): PixelPoint => {
    const delta = subtract(point, palmCenter)
    return {
      x: dot(delta, xAxis) / palmWidth,
      y: dot(delta, yAxis) / palmHeight,
    }
  }

  const upAxis = scale(yAxis, -1)
  const lineJunction = averagePoint([
    offsetPoint(interpolatePoint(indexMcp, thumbMcp, 0.44), upAxis, palmHeight * 0.045),
    offsetPoint(interpolatePoint(indexMcp, middleMcp, 0.18), yAxis, palmHeight * 0.11),
  ])

  return {
    wrist,
    thumbMcp,
    indexMcp,
    middleMcp,
    ringMcp,
    pinkyMcp,
    palmCenter,
    palmWidth,
    palmHeight,
    xAxis,
    yAxis,
    lineJunction,
    localPoint,
    toLocalPoint,
  }
}

const createTemplateLineVariants = (
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
): Record<PalmLineId, PixelPoint[][]> => {
  const { wrist, indexMcp, middleMcp, ringMcp, pinkyMcp, lineJunction, yAxis, palmHeight } =
    createPalmBasis(landmarks, width, height)
  const upAxis = scale(yAxis, -1)
  const towardWrist = (point: PixelPoint, amount: number) => interpolatePoint(point, wrist, amount)
  const between = (from: PixelPoint, to: PixelPoint, amount: number) =>
    interpolatePoint(from, to, amount)
  const lift = (point: PixelPoint, amount: number) => offsetPoint(point, upAxis, palmHeight * amount)

  return {
    heart: [
      [
        lift(interpolatePoint(lineJunction, towardWrist(indexMcp, 0.14), 0.68), 0.01),
        towardWrist(between(indexMcp, middleMcp, 0.52), 0.18),
        towardWrist(middleMcp, 0.19),
        towardWrist(between(middleMcp, ringMcp, 0.54), 0.2),
        towardWrist(ringMcp, 0.22),
        towardWrist(between(ringMcp, pinkyMcp, 0.34), 0.24),
      ],
      [
        lift(interpolatePoint(lineJunction, towardWrist(indexMcp, 0.15), 0.64), 0.006),
        towardWrist(between(indexMcp, middleMcp, 0.5), 0.19),
        towardWrist(middleMcp, 0.21),
        towardWrist(between(middleMcp, ringMcp, 0.56), 0.22),
        towardWrist(ringMcp, 0.235),
        towardWrist(between(ringMcp, pinkyMcp, 0.42), 0.255),
      ],
      [
        lift(interpolatePoint(lineJunction, towardWrist(indexMcp, 0.13), 0.72), 0.014),
        towardWrist(between(indexMcp, middleMcp, 0.56), 0.17),
        towardWrist(middleMcp, 0.18),
        towardWrist(between(middleMcp, ringMcp, 0.5), 0.195),
        towardWrist(ringMcp, 0.21),
        towardWrist(between(ringMcp, pinkyMcp, 0.28), 0.225),
      ],
    ],
    head: [
      [
        lineJunction,
        towardWrist(indexMcp, 0.2),
        towardWrist(between(indexMcp, middleMcp, 0.74), 0.26),
        towardWrist(between(middleMcp, ringMcp, 0.56), 0.34),
        towardWrist(between(ringMcp, pinkyMcp, 0.44), 0.4),
        towardWrist(between(ringMcp, pinkyMcp, 0.74), 0.45),
      ],
      [
        lineJunction,
        towardWrist(indexMcp, 0.215),
        towardWrist(between(indexMcp, middleMcp, 0.7), 0.28),
        towardWrist(between(middleMcp, ringMcp, 0.52), 0.355),
        towardWrist(between(ringMcp, pinkyMcp, 0.38), 0.42),
        towardWrist(between(ringMcp, pinkyMcp, 0.68), 0.47),
      ],
      [
        lineJunction,
        towardWrist(indexMcp, 0.19),
        towardWrist(between(indexMcp, middleMcp, 0.76), 0.24),
        towardWrist(between(middleMcp, ringMcp, 0.58), 0.325),
        towardWrist(between(ringMcp, pinkyMcp, 0.48), 0.385),
        towardWrist(between(ringMcp, pinkyMcp, 0.78), 0.435),
      ],
    ],
    life: [
      [
        lineJunction,
        towardWrist(between(indexMcp, middleMcp, 0.28), 0.24),
        towardWrist(between(indexMcp, middleMcp, 0.42), 0.38),
        towardWrist(between(middleMcp, ringMcp, 0.04), 0.56),
        towardWrist(between(middleMcp, ringMcp, 0.12), 0.75),
      ],
      [
        lineJunction,
        towardWrist(between(indexMcp, middleMcp, 0.24), 0.235),
        towardWrist(between(indexMcp, middleMcp, 0.36), 0.36),
        towardWrist(between(middleMcp, ringMcp, 0.02), 0.54),
        towardWrist(between(middleMcp, ringMcp, 0.08), 0.72),
      ],
      [
        lineJunction,
        towardWrist(between(indexMcp, middleMcp, 0.32), 0.245),
        towardWrist(between(indexMcp, middleMcp, 0.46), 0.4),
        towardWrist(between(middleMcp, ringMcp, 0.06), 0.585),
        towardWrist(between(middleMcp, ringMcp, 0.16), 0.79),
      ],
    ],
  }
}

interface TracedLineCandidate {
  confidence: number
  points: PixelPoint[]
  score: number
}

const traceTemplateCandidate = (
  template: PixelPoint[],
  lineId: PalmLineId,
  texture: PalmTextureData,
  basis: PalmBasis,
): TracedLineCandidate => {
  const denseTemplate = densifyPoints(template, lineId === 'life' ? 42 : 38)
  const traced: PixelPoint[] = []
  let ridgeSum = 0
  let matchedSamples = 0
  let scoreSum = 0

  const getBandPenalty = (candidate: PixelPoint, progress: number) => {
    const local = basis.toLocalPoint(candidate)

    if (lineId === 'heart') {
      const targetU = 0.12 - progress * 0.5
      const targetV = -0.24 + progress * 0.07
      return (
        Math.max(0, Math.abs(local.x - targetU) - 0.16) * 120 +
        Math.max(0, Math.abs(local.y - targetV) - 0.065) * 190 +
        Math.max(0, -0.37 - local.y) * 340 +
        Math.max(0, local.y + 0.05) * 220
      )
    }

    if (lineId === 'head') {
      const targetU = 0.12 - progress * 0.56
      const targetV = -0.14 + progress * 0.28
      return (
        Math.max(0, Math.abs(local.x - targetU) - 0.2) * 105 +
        Math.max(0, Math.abs(local.y - targetV) - 0.1) * 150
      )
    }

    const targetU = 0.08 - progress * 0.02
    const targetV = -0.1 + progress * 0.78
    return (
      Math.max(0, Math.abs(local.x - targetU) - 0.11) * 135 +
      Math.max(0, Math.abs(local.y - targetV) - 0.12) * 155 +
      Math.max(0, -0.2 - local.x) * 180
    )
  }

  denseTemplate.forEach((point, index) => {
    const previous = traced[index - 1]
    const nextTemplate = denseTemplate[Math.min(index + 1, denseTemplate.length - 1)]
    const prevTemplate = denseTemplate[Math.max(index - 1, 0)]
    const tangent = normalizeVector(subtract(nextTemplate, prevTemplate))
    const normal = normalizeVector(perpendicular(tangent))
    const progress = index / Math.max(denseTemplate.length - 1, 1)
    const normalRadius = lineId === 'heart' ? 5 : lineId === 'life' ? 6 : 7
    const tangentRadius = lineId === 'heart' ? 2 : 3
    const continuityWeight = lineId === 'heart' ? 2.8 : lineId === 'life' ? 2.6 : 2.3
    const normalPenaltyWeight = lineId === 'heart' ? 2.5 : lineId === 'life' ? 2.15 : 1.8

    let bestPoint = point
    let bestRidge = 0
    let bestScore = 0

    for (let normalOffset = -normalRadius; normalOffset <= normalRadius; normalOffset += 1) {
      for (let tangentOffset = -tangentRadius; tangentOffset <= tangentRadius; tangentOffset += 1) {
        const candidate = add(
          point,
          add(scale(normal, normalOffset), scale(tangent, tangentOffset * 0.85)),
        )
        const ridge = sampleRidge(texture, candidate)
        if (ridge === 0) {
          continue
        }

        const continuityPenalty = previous ? distance(previous, candidate) * continuityWeight : 0
        const templatePenalty =
          Math.abs(normalOffset) * normalPenaltyWeight + Math.abs(tangentOffset) * 1.05
        const bandPenalty = getBandPenalty(candidate, progress)
        const backwardPenalty =
          previous && candidate.y < previous.y - 2 ? Math.abs(candidate.y - previous.y) * 4.4 : 0
        const score =
          ridge * 1.42 - continuityPenalty - templatePenalty - bandPenalty - backwardPenalty

        if (score > bestScore) {
          bestScore = score
          bestPoint = candidate
          bestRidge = ridge
        }
      }
    }

    ridgeSum += bestRidge
    scoreSum += bestScore
    if (bestRidge >= 24) {
      matchedSamples += 1
    }
    traced.push(bestPoint)
  })

  const coverage = matchedSamples / denseTemplate.length
  const support = ridgeSum / (denseTemplate.length * 190)
  const baseConfidence = clamp(support * 0.62 + coverage * 0.38, 0, 1)
  const averageBandPenalty =
    traced.reduce(
      (sum, point, index) => sum + getBandPenalty(point, index / Math.max(traced.length - 1, 1)),
      0,
    ) / Math.max(traced.length, 1)
  const confidence = clamp(baseConfidence - averageBandPenalty / 260, 0, 1)
  const blendAmount =
    lineId === 'life'
      ? clamp(0.2 + confidence * 0.16, 0.2, 0.36)
      : clamp(0.22 + confidence * 0.24, 0.22, 0.5)

  return {
    confidence,
    score: scoreSum / denseTemplate.length + confidence * 56 + coverage * 34,
    points: chaikinSmoothPoints(
      repeatSmoothPoints(
        traced.map((point, index) => interpolatePoint(denseTemplate[index], point, blendAmount)),
        4,
      ),
      3,
    ),
  }
}

const sampleRidge = (texture: PalmTextureData, point: PixelPoint) => {
  const x = Math.round(((point.x - texture.sourceX) / texture.sourceWidth) * (DETAIL_SIZE - 1))
  const y = Math.round(((point.y - texture.sourceY) / texture.sourceHeight) * (DETAIL_SIZE - 1))

  if (x < 1 || y < 1 || x >= DETAIL_SIZE - 1 || y >= DETAIL_SIZE - 1) {
    return 0
  }

  return texture.ridgeMap[y * DETAIL_SIZE + x]
}

const normalizeLinePoints = (
  points: PixelPoint[],
  width: number,
  height: number,
): Point[] => points.map((point) => toNormalizedPoint(point, width, height))

export const analyzePalmFrame = (
  video: HTMLVideoElement,
  landmarks: NormalizedLandmark[],
  width: number,
  height: number,
): PalmFrameAnalysis | null => {
  const anchorKey = createAnchorKey(landmarks)
  if (
    cachedAnalysis &&
    cachedAnalysis.anchorKey === anchorKey &&
    performance.now() - cachedAnalysis.result.texture.updatedAt < TEXTURE_REFRESH_MS
  ) {
    return cachedAnalysis.result
  }

  const texture = buildPalmTexture(video, landmarks, width, height)
  if (!texture) {
    return null
  }

  const basis = createPalmBasis(landmarks, width, height)
  const templateVariants = createTemplateLineVariants(landmarks, width, height)
  const detectedLines = (Object.keys(templateVariants) as PalmLineId[]).map((lineId) => {
    const bestCandidate = templateVariants[lineId]
      .map((template) => traceTemplateCandidate(template, lineId, texture, basis))
      .reduce((best, current) => (current.score > best.score ? current : best))

    return {
      id: lineId,
      confidence: bestCandidate.confidence,
      points: normalizeLinePoints(bestCandidate.points, width, height),
    }
  })

  const result = {
    texture,
    detectedLines,
  }
  cachedAnalysis = {
    anchorKey,
    result,
  }

  return result
}

const drawSmoothLine = (
  context: CanvasRenderingContext2D,
  points: Point[],
  color: string,
  width: number,
  height: number,
) => {
  if (points.length === 0) {
    return
  }

  const scaledPoints = points.map((point) => ({
    x: point.x * width,
    y: point.y * height,
  }))
  context.beginPath()
  context.moveTo(scaledPoints[0].x, scaledPoints[0].y)

  for (let index = 1; index < scaledPoints.length - 1; index += 1) {
    const current = scaledPoints[index]
    const next = scaledPoints[index + 1]
    const midPoint = {
      x: (current.x + next.x) / 2,
      y: (current.y + next.y) / 2,
    }

    context.quadraticCurveTo(current.x, current.y, midPoint.x, midPoint.y)
  }

  const lastPoint = scaledPoints[scaledPoints.length - 1]
  context.lineTo(lastPoint.x, lastPoint.y)
  context.strokeStyle = color
  context.lineWidth = Math.max(width, height) * 0.006
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.shadowColor = color
  context.shadowBlur = 14
  context.stroke()
  context.shadowBlur = 0
}

const drawLineLabel = (
  context: CanvasRenderingContext2D,
  line: PalmLineInput,
  width: number,
  height: number,
  isMirrored: boolean,
) => {
  const palette = LINE_PALETTE[line.id]
  const rawAnchor = {
    x: line.points[line.points.length - 1].x * width,
    y: line.points[line.points.length - 1].y * height,
  }
  const anchor = isMirrored ? { x: width - rawAnchor.x, y: rawAnchor.y } : rawAnchor
  const labelX = clamp(anchor.x + 10, 18, width - 86)
  const labelY = clamp(anchor.y - 18, 14, height - 34)
  const text = line.id === 'life' ? '生命線' : line.id === 'head' ? '知能線' : '感情線'

  context.save()
  if (isMirrored) {
    context.translate(width, 0)
    context.scale(-1, 1)
  }
  context.font = `600 ${Math.max(12, width * 0.015)}px Inter, "Noto Sans JP", sans-serif`
  const metrics = context.measureText(text)
  const boxWidth = metrics.width + 22
  const boxHeight = 24
  createRoundedRect(context, labelX, labelY, boxWidth, boxHeight, 12)
  context.fillStyle = 'rgba(6, 10, 22, 0.78)'
  context.fill()
  context.strokeStyle = palette.color
  context.lineWidth = 1.2
  context.stroke()
  context.fillStyle = palette.text
  context.fillText(text, labelX + 11, labelY + 16)
  context.restore()
}

const drawPalmTexture = (context: CanvasRenderingContext2D, texture: PalmTextureData) => {
  context.save()
  context.beginPath()
  texture.maskPoints.forEach((point, index) => {
    if (index === 0) {
      context.moveTo(point.x, point.y)
      return
    }

    context.lineTo(point.x, point.y)
  })
  context.closePath()
  context.clip()
  context.globalAlpha = 0.78
  context.drawImage(
    texture.image,
    texture.sourceX,
    texture.sourceY,
    texture.sourceWidth,
    texture.sourceHeight,
  )
  context.restore()
}

export const drawHandOverlay = (
  canvas: HTMLCanvasElement | null,
  analysis: PalmFrameAnalysis | null,
  landmarks: NormalizedLandmark[] | null,
  reading: PalmReading | null,
  isMirrored: boolean,
) => {
  if (!canvas) {
    return
  }

  const context = canvas.getContext('2d')
  if (!context) {
    return
  }

  context.clearRect(0, 0, canvas.width, canvas.height)

  if (!landmarks) {
    return
  }

  context.save()

  const scaledLandmarks = landmarks.map((landmark) =>
    toPixelPoint(landmark, canvas.width, canvas.height),
  )

  if (analysis) {
    drawPalmTexture(context, analysis.texture)
  }

  context.beginPath()
  context.moveTo(scaledLandmarks[0].x, scaledLandmarks[0].y)
  ;[5, 9, 13, 17].forEach((index) => {
    context.lineTo(scaledLandmarks[index].x, scaledLandmarks[index].y)
  })
  context.closePath()
  context.fillStyle = 'rgba(167, 139, 250, 0.1)'
  context.fill()

  context.strokeStyle = 'rgba(196, 181, 253, 0.34)'
  context.lineWidth = Math.max(canvas.width, canvas.height) * 0.0035
  HandLandmarker.HAND_CONNECTIONS.forEach((connection) => {
    const start = scaledLandmarks[connection.start]
    const end = scaledLandmarks[connection.end]
    context.beginPath()
    context.moveTo(start.x, start.y)
    context.lineTo(end.x, end.y)
    context.stroke()
  })

  context.fillStyle = '#f8fafc'
  scaledLandmarks.forEach((point, index) => {
    context.beginPath()
    context.arc(point.x, point.y, index === 0 ? 4.2 : 3.1, 0, Math.PI * 2)
    context.fill()
  })

  if (reading) {
    reading.lines.forEach((line) => {
      drawSmoothLine(context, line.points, LINE_PALETTE[line.id].color, canvas.width, canvas.height)
      drawLineLabel(context, line, canvas.width, canvas.height, isMirrored)
    })
  }

  context.restore()
}
