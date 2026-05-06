import { readFileSync } from 'node:fs'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

type MediaPipeAssetsConfig = {
  assetVersion?: string
}

type PackageConfig = {
  name?: string
}

const mediapipeAssetsConfig = JSON.parse(
  readFileSync(new URL('./mediapipe-assets.config.json', import.meta.url), 'utf8'),
) as MediaPipeAssetsConfig
const packageConfig = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf8'),
) as PackageConfig
const basePath = `/${packageConfig.name ?? 'palm-reading'}/`

export default defineConfig({
  base: basePath,
  plugins: [react()],
  define: {
    __MEDIAPIPE_ASSET_VERSION__: JSON.stringify(mediapipeAssetsConfig.assetVersion ?? 'dev'),
  },
})
