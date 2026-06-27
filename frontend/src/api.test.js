import assert from 'node:assert/strict'
import test from 'node:test'

import { pollBackendHealth } from './api.js'


test('health polling retries until the backend is ready', async () => {
  const states = []
  let attempts = 0
  let health = null
  const controller = new AbortController()

  await pollBackendHealth({
    signal: controller.signal,
    onState: (state) => states.push(state),
    onHealth: (payload) => {
      health = payload
    },
    fetchHealthFn: async () => {
      attempts += 1
      if (attempts < 3) throw new Error('offline')
      return { model_loaded: true }
    },
    retryDelays: [0]
  })

  assert.equal(attempts, 3)
  assert.deepEqual(health, { model_loaded: true })
  assert.equal(states.at(-1), 'ready')
  assert.ok(states.includes('offline'))
  assert.ok(states.includes('waking'))
})


test('aborting health polling stops further retries', async () => {
  const controller = new AbortController()
  let attempts = 0

  await pollBackendHealth({
    signal: controller.signal,
    onState: () => {},
    onHealth: () => {},
    fetchHealthFn: async () => {
      attempts += 1
      controller.abort()
      throw new Error('offline')
    },
    retryDelays: [0]
  })

  assert.equal(attempts, 1)
})
