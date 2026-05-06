import { readFileSync } from 'node:fs'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

type MediaPipeAssetsConfig = {
  assetVersion?: string
}

const mediapipeAssetsConfig = JSON.parse(
  readFileSync(new URL('./mediapipe-assets.config.json', import.meta.url), 'utf8'),
) as MediaPipeAssetsConfig

export default defineConfig({
  base: './',
  plugins: [react()],
  define: {
    __MEDIAPIPE_ASSET_VERSION__: JSON.stringify(mediapipeAssetsConfig.assetVersion ?? 'dev'),
  },
})
