import { access, mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const projectRoot = path.resolve(scriptDir, '..')

const readJsonConfig = async (filePath) => {
  try {
    return JSON.parse(await readFile(filePath, 'utf8'))
  } catch {
    return {}
  }
}

const fileExists = async (filePath) => {
  try {
    await access(filePath)
    return true
  } catch {
    return false
  }
}

const config = await readJsonConfig(path.join(projectRoot, 'mediapipe-assets.config.json'))
const env = process.env

const readInstalledTasksVisionVersion = async () => {
  const packagePath = path.join(
    projectRoot,
    'node_modules',
    '@mediapipe',
    'tasks-vision',
    'package.json',
  )

  try {
    const pkg = JSON.parse(await readFile(packagePath, 'utf8'))
    return typeof pkg.version === 'string' ? pkg.version : '0.10.35'
  } catch {
    return '0.10.35'
  }
}

const tasksVisionVersion = await readInstalledTasksVisionVersion()
const assetVersion = env.MEDIAPIPE_ASSET_VERSION || config.assetVersion || '0.10.35-1'
const releaseTag = env.MEDIAPIPE_RELEASE_TAG || `mediapipe-assets-v${assetVersion}`
const releaseOwner = env.MEDIAPIPE_RELEASE_OWNER || 'yuzuafro'
const releaseRepo = env.MEDIAPIPE_RELEASE_REPO || 'palm-reading'
const assetSource = env.MEDIAPIPE_ASSET_SOURCE || 'auto'
const targetRoot = path.resolve(projectRoot, env.MEDIAPIPE_ASSET_TARGET_DIR || 'public')

const releaseBaseUrl = `https://github.com/${releaseOwner}/${releaseRepo}/releases/download/${releaseTag}`

const assetDefinitions = [
  {
    relativePath: path.join('models', 'hand_landmarker.task'),
    assetFileName: 'hand_landmarker.task',
    officialUrl:
      'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  },
  ...[
    'vision_wasm_internal.js',
    'vision_wasm_internal.wasm',
    'vision_wasm_nosimd_internal.js',
    'vision_wasm_nosimd_internal.wasm',
    'vision_wasm_module_internal.js',
    'vision_wasm_module_internal.wasm',
  ].map((assetFileName) => ({
    relativePath: path.join('wasm', assetFileName),
    assetFileName,
    officialUrl: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${tasksVisionVersion}/wasm/${assetFileName}`,
  })),
]

const downloadOrders = {
  auto: ['release', 'official'],
  release: ['release'],
  official: ['official'],
  local: [],
}

if (!(assetSource in downloadOrders)) {
  throw new Error(
    `Unsupported MEDIAPIPE_ASSET_SOURCE "${assetSource}". Use one of: auto, release, official, local.`,
  )
}

const resolveSourceUrl = (asset, source) =>
  source === 'release' ? `${releaseBaseUrl}/${asset.assetFileName}` : asset.officialUrl

const downloadAsset = async (asset) => {
  const targetPath = path.join(targetRoot, asset.relativePath)
  if (await fileExists(targetPath)) {
    return 'existing'
  }

  if (assetSource === 'local') {
    throw new Error(`Missing local asset: ${path.relative(projectRoot, targetPath)}`)
  }

  await mkdir(path.dirname(targetPath), { recursive: true })

  for (const source of downloadOrders[assetSource]) {
    const url = resolveSourceUrl(asset, source)
    const response = await fetch(url)
    if (!response.ok) {
      continue
    }

    const content = Buffer.from(await response.arrayBuffer())
    await writeFile(targetPath, content)
    return source
  }

  throw new Error(`Unable to prepare asset: ${asset.relativePath}`)
}

const preparedAssets = await Promise.all(
  assetDefinitions.map(async (asset) => ({
    asset,
    result: await downloadAsset(asset),
  })),
)

const missingAssets = await Promise.all(
  assetDefinitions.map(async (asset) => {
    const targetPath = path.join(targetRoot, asset.relativePath)
    return (await fileExists(targetPath)) ? null : asset.relativePath
  }),
)

const unresolvedAssets = missingAssets.filter(Boolean)
if (unresolvedAssets.length > 0) {
  throw new Error(`Assets are still missing: ${unresolvedAssets.join(', ')}`)
}

const summary = preparedAssets.reduce(
  (counts, { result }) => {
    counts[result] = (counts[result] || 0) + 1
    return counts
  },
  /** @type {Record<string, number>} */ ({}),
)

console.log(
  `Prepared MediaPipe assets (version ${assetVersion}) in ${path.relative(projectRoot, targetRoot) || '.'}: ${Object.entries(summary)
    .map(([key, value]) => `${key}=${value}`)
    .join(', ')}`,
)
