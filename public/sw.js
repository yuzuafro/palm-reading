const ASSET_VERSION = new URL(self.location.href).searchParams.get('v') ?? 'dev'
const CACHE_NAME = `palm-reading-shell-${ASSET_VERSION}`
const BASE_PATH = new URL(self.registration.scope).pathname
const PRECACHE_URLS = [
  BASE_PATH,
  `${BASE_PATH}manifest.webmanifest`,
  `${BASE_PATH}favicon.svg`,
  `${BASE_PATH}apple-touch-icon.png`,
  `${BASE_PATH}icon-192.png`,
  `${BASE_PATH}icon-512.png`,
  `${BASE_PATH}models/hand_landmarker.task`,
  `${BASE_PATH}wasm/vision_wasm_internal.js`,
  `${BASE_PATH}wasm/vision_wasm_internal.wasm`,
  `${BASE_PATH}wasm/vision_wasm_nosimd_internal.js`,
  `${BASE_PATH}wasm/vision_wasm_nosimd_internal.wasm`,
  `${BASE_PATH}wasm/vision_wasm_module_internal.js`,
  `${BASE_PATH}wasm/vision_wasm_module_internal.wasm`,
]

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting()),
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))),
      )
      .then(() => self.clients.claim()),
  )
})

self.addEventListener('fetch', (event) => {
  const { request } = event
  if (request.method !== 'GET') {
    return
  }

  const url = new URL(request.url)
  if (url.origin !== self.location.origin) {
    return
  }

  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const responseClone = response.clone()
          caches.open(CACHE_NAME).then((cache) => cache.put(BASE_PATH, responseClone))
          return response
        })
        .catch(async () => {
          const cachedDocument = await caches.match(request)
          return cachedDocument || caches.match(BASE_PATH)
        }),
    )
    return
  }

  event.respondWith(
    caches.match(request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse
      }

      return fetch(request).then((response) => {
        if (response.ok) {
          const responseClone = response.clone()
          caches.open(CACHE_NAME).then((cache) => cache.put(request, responseClone))
        }

        return response
      })
    }),
  )
})
