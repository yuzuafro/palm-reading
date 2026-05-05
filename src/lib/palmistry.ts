import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import type { PalmLineId } from './linePalette'

export interface Point {
  x: number
  y: number
}

export interface PalmLine {
  id: PalmLineId
  label: string
  tone: string
  title: string
  summary: string
  points: Point[]
}

export interface PalmLineInput {
  id: PalmLineId
  points: Point[]
  confidence?: number
}

export interface PalmTrait {
  label: string
  value: string
  detail: string
}

export interface PalmReading {
  handLabel: string
  headline: string
  summary: string
  guidance: string
  lines: PalmLine[]
  traits: PalmTrait[]
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))
const RELIABLE_DETECTION_THRESHOLDS: Record<PalmLineId, number> = {
  life: 0.34,
  head: 0.32,
  heart: 0.35,
}

const normalize = (value: number, min: number, max: number) => {
  if (max === min) {
    return 0
  }

  return clamp((value - min) / (max - min), 0, 1)
}

const distance = (from: Point, to: Point) => Math.hypot(to.x - from.x, to.y - from.y)

const average = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length

const mix = (from: Point, to: Point, amount: number): Point => ({
  x: from.x + (to.x - from.x) * amount,
  y: from.y + (to.y - from.y) * amount,
})

const offset = (point: Point, direction: Point, strength: number): Point => ({
  x: point.x + direction.x * strength,
  y: point.y + direction.y * strength,
})

const averagePoint = (points: Point[]): Point => ({
  x: average(points.map((point) => point.x)),
  y: average(points.map((point) => point.y)),
})

const direction = (from: Point, to: Point): Point => {
  const length = distance(from, to) || 1
  return {
    x: (to.x - from.x) / length,
    y: (to.y - from.y) / length,
  }
}

const angleAt = (from: Point, vertex: Point, to: Point) => {
  const first = { x: from.x - vertex.x, y: from.y - vertex.y }
  const second = { x: to.x - vertex.x, y: to.y - vertex.y }
  const firstLength = Math.hypot(first.x, first.y) || 1
  const secondLength = Math.hypot(second.x, second.y) || 1
  const cosine =
    (first.x * second.x + first.y * second.y) / (firstLength * secondLength)

  return Math.acos(clamp(cosine, -1, 1))
}

const lineLength = (points: Point[]) =>
  points.slice(1).reduce((sum, point, index) => sum + distance(points[index], point), 0)

const getLineMetric = (points: Point[]) => {
  if (points.length === 0) {
    return {
      length: 0,
      xSpan: 0,
      ySpan: 0,
      start: { x: 0, y: 0 },
      end: { x: 0, y: 0 },
      averageY: 0,
    }
  }

  const xs = points.map((point) => point.x)
  const ys = points.map((point) => point.y)

  return {
    length: lineLength(points),
    xSpan: Math.max(...xs) - Math.min(...xs),
    ySpan: Math.max(...ys) - Math.min(...ys),
    start: points[0],
    end: points[points.length - 1],
    averageY: average(ys),
  }
}

const resampleLine = (points: Point[], sampleCount = 24): Point[] => {
  if (points.length <= 1) {
    return points
  }

  const lengths = [0]
  for (let index = 1; index < points.length; index += 1) {
    lengths.push(lengths[index - 1] + distance(points[index - 1], points[index]))
  }

  const totalLength = lengths[lengths.length - 1]
  if (totalLength === 0) {
    return Array.from({ length: sampleCount }, () => points[0])
  }

  return Array.from({ length: sampleCount }, (_, sampleIndex) => {
    const targetLength = (sampleIndex / (sampleCount - 1)) * totalLength
    let segmentIndex = 1
    while (segmentIndex < lengths.length && lengths[segmentIndex] < targetLength) {
      segmentIndex += 1
    }

    const startIndex = Math.max(0, segmentIndex - 1)
    const endIndex = Math.min(points.length - 1, segmentIndex)
    const segmentLength = lengths[endIndex] - lengths[startIndex]
    const amount =
      segmentLength === 0 ? 0 : (targetLength - lengths[startIndex]) / Math.max(segmentLength, 1e-6)

    return mix(points[startIndex], points[endIndex], amount)
  })
}

const blendLinePoints = (fallbackPoints: Point[], detectedPoints: Point[], amount: number): Point[] => {
  const sampleCount = Math.max(fallbackPoints.length, detectedPoints.length, 18)
  const fallback = resampleLine(fallbackPoints, sampleCount)
  const detected = resampleLine(detectedPoints, sampleCount)
  return fallback.map((point, index) => mix(point, detected[index], amount))
}

const composeLinePoints = (
  lineId: PalmLineId,
  fallbackPoints: Point[],
  detectedLine?: PalmLineInput,
): Point[] => {
  if (!detectedLine?.points.length) {
    return fallbackPoints
  }

  const confidence = detectedLine.confidence ?? 0
  const blendFloor = lineId === 'heart' ? 0.14 : 0.12
  if (confidence < blendFloor) {
    return fallbackPoints
  }

  const threshold = RELIABLE_DETECTION_THRESHOLDS[lineId]
  const normalizedConfidence = clamp(
    (confidence - blendFloor) / Math.max(threshold - blendFloor, 0.01),
    0,
    1,
  )
  const maxBlend = lineId === 'life' ? 0.82 : 0.88
  const blendAmount = clamp(0.22 + normalizedConfidence * (maxBlend - 0.22), 0.22, maxBlend)
  return blendLinePoints(fallbackPoints, detectedLine.points, blendAmount)
}

const describeBand = (
  score: number,
  high: { tone: string; title: string; summary: string },
  middle: { tone: string; title: string; summary: string },
  low: { tone: string; title: string; summary: string },
) => {
  if (score >= 0.67) {
    return high
  }

  if (score >= 0.38) {
    return middle
  }

  return low
}

const formatHandLabel = (label?: string) => {
  if (label?.toLowerCase().includes('left')) {
    return '左手'
  }

  if (label?.toLowerCase().includes('right')) {
    return '右手'
  }

  return '検出中の手'
}

export function analyzePalm(
  landmarks: NormalizedLandmark[],
  handednessLabel?: string,
  detectedLines: PalmLineInput[] = [],
): PalmReading {
  const wrist = landmarks[0]
  const thumbMcp = landmarks[2]
  const thumbTip = landmarks[4]
  const indexMcp = landmarks[5]
  const indexTip = landmarks[8]
  const middleMcp = landmarks[9]
  const middleTip = landmarks[12]
  const ringMcp = landmarks[13]
  const ringTip = landmarks[16]
  const pinkyMcp = landmarks[17]
  const pinkyTip = landmarks[20]

  const palmWidth = distance(indexMcp, pinkyMcp)
  const palmLength = distance(wrist, middleMcp)
  const fingerSpread =
    (distance(indexTip, middleTip) +
      distance(middleTip, ringTip) +
      distance(ringTip, pinkyTip)) /
    palmWidth
  const fingerLengthAverage = average([
    distance(indexMcp, indexTip),
    distance(middleMcp, middleTip),
    distance(ringMcp, ringTip),
    distance(pinkyMcp, pinkyTip),
  ])
  const thumbAngle = angleAt(thumbTip, wrist, indexMcp)
  const thumbReach = distance(thumbTip, indexTip) / palmWidth
  const palmRatio = palmLength / palmWidth
  const fingerBalance =
    1 - Math.abs(distance(indexMcp, indexTip) - distance(ringMcp, ringTip)) / fingerLengthAverage

  const detectedLineMap = Object.fromEntries(
    detectedLines.map((line) => [line.id, line]),
  ) as Partial<Record<PalmLineId, PalmLineInput>>
  const metricPoints = (lineId: PalmLineId) =>
    (detectedLineMap[lineId]?.confidence ?? 0) >= 0.18 ? detectedLineMap[lineId]?.points ?? [] : []
  const lifeMetrics = getLineMetric(metricPoints('life'))
  const headMetrics = getLineMetric(metricPoints('head'))
  const heartMetrics = getLineMetric(metricPoints('heart'))
  const palmTopCenter = averagePoint([indexMcp, middleMcp, ringMcp, pinkyMcp])

  const lifeScore = clamp(
    normalize(thumbAngle, 0.45, 1.35) * 0.5 +
      normalize(thumbReach, 0.75, 1.7) * 0.3 +
      normalize(palmRatio, 0.8, 1.35) * 0.2 +
      normalize(lifeMetrics.length / palmLength, 0.45, 0.95) * 0.25,
    0,
    1,
  )
  const headScore = clamp(
    normalize(fingerLengthAverage / palmLength, 0.62, 1.14) * 0.45 +
      normalize(1 - fingerSpread, 0.05, 0.72) * 0.35 +
      normalize(fingerBalance, 0.4, 1) * 0.2 +
      normalize(headMetrics.xSpan / palmWidth, 0.3, 0.82) * 0.22,
    0,
    1,
  )
  const heartScore = clamp(
    normalize(fingerSpread, 0.32, 1.16) * 0.55 +
      normalize(fingerBalance, 0.4, 1) * 0.2 +
      normalize((pinkyTip.y - indexTip.y) * -1, -0.22, 0.24) * 0.25 +
      normalize((palmTopCenter.y - heartMetrics.averageY) / palmLength, -0.08, 0.26) * 0.28,
    0,
    1,
  )

  const palmDirection = direction(
    {
      x: average([indexMcp.x, middleMcp.x, ringMcp.x, pinkyMcp.x]),
      y: average([indexMcp.y, middleMcp.y, ringMcp.y, pinkyMcp.y]),
    },
    wrist,
  )
  const wristDirection = direction(palmTopCenter, wrist)
  const upDirection = { x: -wristDirection.x, y: -wristDirection.y }
  const lineJunction = averagePoint([
    offset(mix(indexMcp, thumbMcp, 0.44), upDirection, palmLength * 0.02),
    offset(mix(indexMcp, middleMcp, 0.18), palmDirection, palmLength * 0.06),
  ])

  const lifeDescription = describeBand(
    lifeScore,
    {
      tone: 'エネルギッシュ',
      title: '生命線は大きく弧を描くタイプ',
      summary: '行動の切り替えが早く、環境が動くほど持ち味が出やすい傾向です。',
    },
    {
      tone: 'バランス型',
      title: '生命線は安定して伸びるタイプ',
      summary: '派手さより継続力が強み。自分のペースを守るほど安定して力を出せます。',
    },
    {
      tone: '慎重派',
      title: '生命線はコンパクトにまとまるタイプ',
      summary: '無理に前へ出るより、準備を整えてから動くと成果につながりやすい流れです。',
    },
  )

  const headDescription = describeBand(
    headScore,
    {
      tone: '集中型',
      title: '知能線は長めで深掘り志向',
      summary: '一度テーマを決めると集中して掘り下げるタイプ。設計や検証に強みがあります。',
    },
    {
      tone: '柔軟型',
      title: '知能線はしなやかでバランス良好',
      summary: 'ロジックと直感のバランスが良く、状況に応じて考え方を切り替えられます。',
    },
    {
      tone: '発想型',
      title: '知能線はひらめき優先タイプ',
      summary: '枠に収まらない発想が魅力。まず形にしてから改善していく進め方と相性が良いです。',
    },
  )

  const heartDescription = describeBand(
    heartScore,
    {
      tone: 'オープン',
      title: '感情線は高めで表現がストレート',
      summary: '気持ちを言葉や行動にしやすく、人との距離を縮めるのが得意な傾向です。',
    },
    {
      tone: '安定型',
      title: '感情線は穏やかで扱いやすいタイプ',
      summary: '熱量と冷静さのバランスが良く、関係性を丁寧に育てるスタイルです。',
    },
    {
      tone: '内省型',
      title: '感情線は慎重で観察寄り',
      summary: '相手をよく見てから心を開くタイプ。信頼関係ができると一気に深くつながれます。',
    },
  )

  const handLabel = formatHandLabel(handednessLabel)
  const dominantLine = [
    { label: '生命線', score: lifeScore, tone: lifeDescription.tone },
    { label: '知能線', score: headScore, tone: headDescription.tone },
    { label: '感情線', score: heartScore, tone: heartDescription.tone },
  ].sort((left, right) => right.score - left.score)[0]

  const fallbackLifeLine: PalmLine = {
    id: 'life',
    label: '生命線',
    tone: lifeDescription.tone,
    title: lifeDescription.title,
    summary: lifeDescription.summary,
    points: [
      lineJunction,
      mix(mix(indexMcp, middleMcp, 0.28), wrist, 0.24),
      mix(mix(indexMcp, middleMcp, 0.42), wrist, 0.38),
      mix(mix(middleMcp, ringMcp, 0.04), wrist, 0.56),
      mix(mix(middleMcp, ringMcp, 0.12), wrist, 0.76),
    ],
  }

  const fallbackHeadLine: PalmLine = {
    id: 'head',
    label: '知能線',
    tone: headDescription.tone,
    title: headDescription.title,
    summary: headDescription.summary,
    points: [
      lineJunction,
      mix(indexMcp, wrist, 0.2),
      mix(mix(indexMcp, middleMcp, 0.74), wrist, 0.26),
      mix(mix(middleMcp, ringMcp, 0.56), wrist, 0.34),
      mix(mix(ringMcp, pinkyMcp, 0.44), wrist, 0.4),
      mix(mix(ringMcp, pinkyMcp, 0.74), wrist, 0.45),
    ],
  }

  const fallbackHeartLine: PalmLine = {
    id: 'heart',
    label: '感情線',
    tone: heartDescription.tone,
    title: heartDescription.title,
    summary: heartDescription.summary,
    points: [
      mix(lineJunction, mix(indexMcp, wrist, 0.14), 0.68),
      mix(mix(indexMcp, middleMcp, 0.52), wrist, 0.18),
      mix(middleMcp, wrist, 0.19),
      mix(mix(middleMcp, ringMcp, 0.54), wrist, 0.2),
      mix(ringMcp, wrist, 0.22),
      mix(mix(ringMcp, pinkyMcp, 0.34), wrist, 0.24),
    ],
  }

  const lifeLine: PalmLine = {
    ...fallbackLifeLine,
    points: composeLinePoints('life', fallbackLifeLine.points, detectedLineMap.life),
  }
  const headLine: PalmLine = {
    ...fallbackHeadLine,
    points: composeLinePoints('head', fallbackHeadLine.points, detectedLineMap.head),
  }
  const heartLine: PalmLine = {
    ...fallbackHeartLine,
    points: composeLinePoints('heart', fallbackHeartLine.points, detectedLineMap.heart),
  }

  return {
    handLabel,
    headline: `${dominantLine.label}が目立つ ${dominantLine.tone}タイプ`,
    summary: `${lifeDescription.tone}な生命線、${headDescription.tone}な知能線、${heartDescription.tone}な感情線が見えています。今は「勢い」と「判断」のバランスが取りやすい手つきです。`,
    guidance:
      dominantLine.label === '生命線'
        ? 'まず動いてから微調整する進め方が吉。新しい挑戦を小さく試すと流れが伸びます。'
        : dominantLine.label === '知能線'
          ? '情報を集めて整理すると運気が整いやすいタイミングです。メモ化や見える化が相性良好です。'
          : '人との会話や共有から運が開きやすい流れです。感謝や好意を早めに伝えると巡りが良くなります。',
    lines: [lifeLine, headLine, heartLine],
    traits: [
      {
        label: '行動テンポ',
        value: lifeScore >= 0.67 ? '前のめりに進める' : lifeScore >= 0.38 ? '安定して積み上げる' : '慎重に見極める',
        detail: `親指の開きと手の比率から見ると、今は${lifeDescription.tone}モードです。`,
      },
      {
        label: '思考モード',
        value: headScore >= 0.67 ? '深く集中する' : headScore >= 0.38 ? '柔軟に切り替える' : '直感でひらめく',
        detail: `指の長さと広がりから、${headDescription.tone}な考え方が出やすく見えます。`,
      },
      {
        label: '対人スタンス',
        value: heartScore >= 0.67 ? '感情を伝えやすい' : heartScore >= 0.38 ? '穏やかに寄り添う' : '距離感を大事にする',
        detail: `指先の広がり方から、${heartDescription.tone}なコミュニケーション傾向が出ています。`,
      },
    ],
  }
}
