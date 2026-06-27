const API_BASE_URL = (import.meta.env?.VITE_API_BASE_URL || '').replace(/\/+$/, '')
const DEFAULT_TIMEOUT_MS = 30_000
const HEALTH_RETRY_DELAYS_MS = [2_000, 4_000, 8_000, 15_000]

export class ApiError extends Error {
  constructor(message, status = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function apiUrl(path) {
  return `${API_BASE_URL}${path}`
}

async function request(path, options = {}) {
  const timeoutController = new AbortController()
  const timeoutId = setTimeout(() => timeoutController.abort(), options.timeoutMs || DEFAULT_TIMEOUT_MS)
  const signal = options.signal
    ? AbortSignal.any([options.signal, timeoutController.signal])
    : timeoutController.signal

  try {
    return await fetch(apiUrl(path), { ...options, signal })
  } catch (error) {
    if (timeoutController.signal.aborted) {
      throw new ApiError('The backend did not respond in time.')
    }
    throw error
  } finally {
    clearTimeout(timeoutId)
  }
}

export async function predictOctImage(file) {
  const body = new FormData()
  body.append('image', file)

  const response = await request('/api/predict', {
    method: 'POST',
    body
  })

  const payload = await response.json().catch(() => ({}))

  if (!response.ok) {
    const message = payload.detail || 'Inference failed. Please try another image.'
    throw new ApiError(message, response.status)
  }

  return payload
}

export async function fetchHealth({ signal } = {}) {
  const response = await request('/api/health', { signal, timeoutMs: DEFAULT_TIMEOUT_MS })
  if (!response.ok) {
    throw new ApiError('Backend health check failed.', response.status)
  }
  const payload = await response.json()
  if (payload.model_loaded !== true) {
    throw new ApiError('Backend model is unavailable.', 503)
  }
  return payload
}

export async function pollBackendHealth({
  signal,
  onState,
  onHealth,
  fetchHealthFn = fetchHealth,
  retryDelays = HEALTH_RETRY_DELAYS_MS
}) {
  let attempt = 0
  onState('checking')

  while (!signal.aborted) {
    if (attempt > 0) onState('waking')

    try {
      const payload = await fetchHealthFn({ signal })
      if (signal.aborted) return
      onHealth(payload)
      onState('ready')
      return
    } catch (error) {
      if (signal.aborted || error?.name === 'AbortError') return
      onState('offline')
    }

    const delay = retryDelays[Math.min(attempt, retryDelays.length - 1)]
    attempt += 1
    await wait(delay, signal)
  }
}

function wait(delay, signal) {
  return new Promise((resolve) => {
    const timeoutId = setTimeout(resolve, delay)
    signal.addEventListener(
      'abort',
      () => {
        clearTimeout(timeoutId)
        resolve()
      },
      { once: true }
    )
  })
}
