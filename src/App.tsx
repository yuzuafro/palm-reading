import { useCallback, useEffect, useRef, useState, type CSSProperties } from 'react'
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from '@mediapipe/tasks-vision'
import './App.css'
import { analyzePalmFrame, drawHandOverlay } from './lib/draw'
import { LINE_PALETTE } from './lib/linePalette'
import { analyzePalm, type PalmReading } from './lib/palmistry'

type SessionState = 'idle' | 'loading' | 'running' | 'paused' | 'error'
type StopSessionOptions = {
  resetState?: boolean
  preserveReading?: boolean
  preserveOverlay?: boolean
}

const WASM_PATH = new URL(`${import.meta.env.BASE_URL}wasm`, window.location.href).toString()
const MODEL_PATH = new URL(
  `${import.meta.env.BASE_URL}models/hand_landmarker.task`,
  window.location.href,
).toString()

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const handLandmarkerRef = useRef<HandLandmarker | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const frameRef = useRef<number | null>(null)
  const uiThrottleRef = useRef(0)
  const lastHandSeenRef = useRef(0)

  const [sessionState, setSessionState] = useState<SessionState>('idle')
  const [feedback, setFeedback] = useState('カメラを開始すると手のひらを読み取れます。')
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [reading, setReading] = useState<PalmReading | null>(null)
  const [isHandDetected, setIsHandDetected] = useState(false)
  const [installPromptEvent, setInstallPromptEvent] = useState<BeforeInstallPromptEvent | null>(
    null,
  )

  const isSecure = window.isSecureContext || window.location.hostname === 'localhost'

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) {
      return
    }

    const context = canvas.getContext('2d')
    if (!context) {
      return
    }

    context.clearRect(0, 0, canvas.width, canvas.height)
  }, [])

  const freezeCanvas = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || canvas.width === 0 || canvas.height === 0) {
      return
    }

    const context = canvas.getContext('2d')
    if (!context) {
      return
    }

    const overlay = context.getImageData(0, 0, canvas.width, canvas.height)
    context.clearRect(0, 0, canvas.width, canvas.height)
    context.drawImage(video, 0, 0, canvas.width, canvas.height)
    context.putImageData(overlay, 0, 0)
  }, [])

  const stopSession = useCallback(
    ({
      resetState = true,
      preserveReading = false,
      preserveOverlay = false,
    }: StopSessionOptions = {}) => {
      if (preserveOverlay) {
        freezeCanvas()
      }

      if (frameRef.current !== null) {
        window.cancelAnimationFrame(frameRef.current)
        frameRef.current = null
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
        streamRef.current = null
      }

      const video = videoRef.current
      if (video) {
        video.pause()
        video.srcObject = null
      }

      if (!preserveOverlay) {
        clearCanvas()
      }

      if (resetState) {
        setSessionState(preserveReading ? 'paused' : 'idle')
        setFeedback(
          preserveReading
            ? '停止中です。最後に検出した手相結果を保持しています。'
            : 'カメラを開始すると手のひらを読み取れます。',
        )
        setErrorMessage(null)
        setIsHandDetected(false)
        if (!preserveReading) {
          setReading(null)
        }
      }
    },
    [clearCanvas, freezeCanvas],
  )

  const syncCanvasSize = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current

    if (!video || !canvas || video.videoWidth === 0 || video.videoHeight === 0) {
      return false
    }

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
    }

    return true
  }, [])

  const updateUi = useCallback((nextReading: PalmReading | null, nextFeedback: string) => {
    const now = performance.now()
    if (now - uiThrottleRef.current < 220) {
      return
    }

    uiThrottleRef.current = now
    setReading(nextReading)
    setFeedback(nextFeedback)
  }, [])

  const renderFrame = useCallback(
    (result: HandLandmarkerResult) => {
      const landmarks = result.landmarks[0]
      setIsHandDetected(Boolean(landmarks))
      const handednessLabel =
        result.handedness[0]?.[0]?.displayName ?? result.handedness[0]?.[0]?.categoryName
      const video = videoRef.current
      const canvas = canvasRef.current
      const frameAnalysis =
        landmarks && video && canvas
          ? analyzePalmFrame(video, landmarks, canvas.width, canvas.height)
          : null
      const nextReading =
        landmarks && frameAnalysis
          ? analyzePalm(landmarks, handednessLabel, frameAnalysis.detectedLines)
          : null

      drawHandOverlay(canvasRef.current, frameAnalysis, landmarks ?? null, nextReading)

      if (nextReading) {
        lastHandSeenRef.current = performance.now()
        updateUi(
          nextReading,
          `${nextReading.handLabel}を読み取り中です。`,
        )
        return
      }

      if (performance.now() - lastHandSeenRef.current > 650) {
        updateUi(null, '手のひら全体を画面に収めてください。')
      }
    },
    [updateUi],
  )

  const startSession = useCallback(
    async () => {
      if (!isSecure) {
        setSessionState('error')
        setErrorMessage('カメラ利用には HTTPS もしくは localhost が必要です。')
        return
      }

      if (!navigator.mediaDevices?.getUserMedia) {
        setSessionState('error')
        setErrorMessage('このブラウザではカメラ API を利用できません。')
        return
      }

      stopSession({ resetState: false })
      setSessionState('loading')
      setErrorMessage(null)
      setFeedback('手検出モデルを読み込んでいます...')

      try {
        if (!handLandmarkerRef.current) {
          const vision = await FilesetResolver.forVisionTasks(WASM_PATH)
          handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: MODEL_PATH,
            },
            runningMode: 'VIDEO',
            numHands: 1,
            minHandDetectionConfidence: 0.55,
            minHandPresenceConfidence: 0.55,
            minTrackingConfidence: 0.55,
          })
        } else {
          await handLandmarkerRef.current.setOptions({ runningMode: 'VIDEO', numHands: 1 })
        }

        setFeedback('カメラを起動しています...')

        let stream: MediaStream
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
              facingMode: { ideal: 'user' },
              width: { ideal: 1280 },
              height: { ideal: 720 },
            },
          })
        } catch {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: true,
          })
        }

        streamRef.current = stream

        const video = videoRef.current
        if (!video) {
          throw new Error('映像要素を初期化できませんでした。')
        }

        video.srcObject = stream
        await video.play()
        syncCanvasSize()

        setSessionState('running')
        setFeedback('手のひらを画面に向けてください。')

        const loop = () => {
          const currentVideo = videoRef.current
          const handLandmarker = handLandmarkerRef.current

          if (!currentVideo || !handLandmarker) {
            return
          }

          if (!syncCanvasSize() || currentVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            frameRef.current = window.requestAnimationFrame(loop)
            return
          }

          const result = handLandmarker.detectForVideo(currentVideo, performance.now())
          renderFrame(result)
          frameRef.current = window.requestAnimationFrame(loop)
        }

        frameRef.current = window.requestAnimationFrame(loop)
      } catch (error) {
        stopSession({ resetState: false })
        setSessionState('error')
        setReading(null)
        setErrorMessage(
          error instanceof Error
            ? error.message
            : 'カメラまたは手検出モデルの初期化に失敗しました。',
        )
      }
    },
    [isSecure, renderFrame, stopSession, syncCanvasSize],
  )

  const handleInstall = useCallback(async () => {
    if (!installPromptEvent) {
      setFeedback('ブラウザのメニューからホーム画面に追加できます。')
      return
    }

    await installPromptEvent.prompt()
    const choice = await installPromptEvent.userChoice
    setInstallPromptEvent(null)
    setFeedback(
      choice.outcome === 'accepted'
        ? 'ホーム画面に追加しました。'
        : 'あとからブラウザのメニューから追加できます。',
    )
  }, [installPromptEvent])

  useEffect(() => {
    const handleBeforeInstallPrompt = (event: Event) => {
      event.preventDefault()
      setInstallPromptEvent(event as BeforeInstallPromptEvent)
    }

    window.addEventListener('beforeinstallprompt', handleBeforeInstallPrompt)
    return () => {
      window.removeEventListener('beforeinstallprompt', handleBeforeInstallPrompt)
    }
  }, [])

  useEffect(() => {
    return () => {
      stopSession({ resetState: false })
      handLandmarkerRef.current?.close()
      handLandmarkerRef.current = null
    }
  }, [stopSession])

  const statusLabel =
    sessionState === 'running'
      ? '解析中'
      : sessionState === 'loading'
        ? '準備中'
        : sessionState === 'paused'
          ? '停止中'
        : sessionState === 'error'
          ? 'エラー'
          : '待機中'

  return (
    <div className="app-shell">
      <header className="hero-panel">
        <div className="hero-copy">
          <h1>手相占い</h1>
          <p className="lead">
            カメラで手のひらを読み取り、生命線・知能線・感情線を重ねて表示します。
          </p>
          <div className="hero-actions">
            <button
              type="button"
              className="primary-button"
              onClick={() => void startSession()}
              disabled={sessionState === 'loading'}
            >
              {sessionState === 'running'
                ? '再スキャン'
                : sessionState === 'paused'
                  ? '再開'
                  : '開始'}
            </button>
            <button
              type="button"
              className="secondary-button"
              onClick={() => stopSession({ preserveReading: true, preserveOverlay: true })}
              disabled={sessionState !== 'running'}
            >
              停止
            </button>
            <button type="button" className="secondary-button" onClick={() => void handleInstall()}>
              ホーム画面に追加
            </button>
          </div>
          <div className="meta-row">
            <span className={`status-pill status-${sessionState}`}>{statusLabel}</span>
          </div>
        </div>
      </header>

      <main className="content-grid">
        <section className="camera-card">
          <div className="card-header camera-header">
            <div>
              <h2>プレビュー</h2>
            </div>
            <p className="card-caption">{feedback}</p>
          </div>

          <div className="camera-stage">
            <video ref={videoRef} className="camera-video" playsInline muted />
            <canvas ref={canvasRef} className="camera-overlay" />
            <div
              className={`stage-overlay${
                isHandDetected || sessionState !== 'running' ? ' stage-overlay-hidden' : ''
              }`}
            >
              <span>手のひら全体をフレーム中央へ</span>
            </div>
          </div>

          <div className="line-legend" aria-label="線の色の凡例">
            {Object.entries(LINE_PALETTE).map(([lineId, palette]) => (
              <div key={lineId} className="legend-chip">
                <span
                  className="legend-swatch"
                  style={{ backgroundColor: palette.color, boxShadow: `0 0 18px ${palette.glow}` }}
                />
                <span>{lineId === 'life' ? '生命線' : lineId === 'head' ? '知能線' : '感情線'}</span>
              </div>
            ))}
          </div>

          {errorMessage ? <p className="error-banner">{errorMessage}</p> : null}
          {!isSecure ? (
            <p className="warning-banner">
              カメラ API は HTTPS か localhost でのみ利用できます。
            </p>
          ) : null}
        </section>

        <aside className="results-card">
          <div className="card-header">
            <div>
              <h2>診断結果</h2>
            </div>
          </div>

          {reading ? (
            <>
              <section className="summary-panel">
                <p className="summary-eyebrow">{reading.handLabel}</p>
                <h3>{reading.headline}</h3>
                <p>{reading.summary}</p>
                <p className="guidance">{reading.guidance}</p>
              </section>

                <section className="line-section">
                  {reading.lines.map((line) => (
                   <article
                     key={line.id}
                     className={`line-card line-${line.id}`}
                      style={
                        {
                          borderColor: `${LINE_PALETTE[line.id].color}66`,
                          boxShadow: `inset 0 0 0 1px ${LINE_PALETTE[line.id].soft}, 0 20px 60px rgba(0, 0, 0, 0.28)`,
                        } as CSSProperties
                      }
                    >
                     <div className="line-card-head">
                       <h3 style={{ color: LINE_PALETTE[line.id].text }}>
                         <span
                           className="line-dot"
                           style={{
                             backgroundColor: LINE_PALETTE[line.id].color,
                             boxShadow: `0 0 16px ${LINE_PALETTE[line.id].glow}`,
                           }}
                         />
                         {line.label}
                       </h3>
                       <span
                         style={{
                           background: LINE_PALETTE[line.id].soft,
                           color: LINE_PALETTE[line.id].text,
                           borderColor: `${LINE_PALETTE[line.id].color}55`,
                         }}
                       >
                         {line.tone}
                       </span>
                     </div>
                     <p className="line-title" style={{ color: LINE_PALETTE[line.id].text }}>
                       {line.title}
                     </p>
                     <p>{line.summary}</p>
                   </article>
                 ))}
              </section>

              <section className="traits-section">
                {reading.traits.map((trait) => (
                  <article key={trait.label} className="trait-card">
                    <p className="trait-label">{trait.label}</p>
                    <h3>{trait.value}</h3>
                    <p>{trait.detail}</p>
                  </article>
                ))}
              </section>
            </>
          ) : (
            <section className="empty-panel">
              <h3>まだ手のひらを検出していません</h3>
              <p>カメラを開始して、手のひらを画面にまっすぐ向けてみてください。</p>
              <p>読み取りが始まると、3 本の線を画面に重ねて表示します。</p>
            </section>
          )}
        </aside>
      </main>
    </div>
  )
}

export default App
