<template>
  <main class="shell">
    <section class="workspace">
      <div class="topbar">
        <div class="hero-copy">
          <div class="intro">
            <p class="eyebrow">Retinal OCT inference</p>
            <h1>Disease scan workspace</h1>
            <p class="subtitle">
              Upload one retinal OCT image to predict its disease class with the ResNet50 OCT model.
            </p>
          </div>
        </div>

        <div class="quick-status" aria-label="System status">
          <span class="status-title">System profile</span>
          <span class="quick-pill">oct_resnet50.keras</span>
          <span class="quick-pill">224px RGB</span>
          <span class="quick-pill">Image only</span>
        </div>
      </div>

      <div class="tool-grid">
        <section class="panel upload-panel" aria-label="Image upload">
          <div class="panel-heading">
            <div>
              <span class="section-kicker">Input</span>
              <h2>OCT image</h2>
            </div>
            <span class="mode-badge">Image only</span>
          </div>

          <label
            class="drop-zone"
            :class="{ 'is-dragging': isDragging, 'has-image': previewUrl }"
            @dragover.prevent="isDragging = true"
            @dragleave.prevent="isDragging = false"
            @drop.prevent="handleDrop"
          >
            <input
              ref="fileInput"
              type="file"
              accept="image/*"
              class="file-input"
              @change="handleFileChange"
            />

            <img v-if="previewUrl" :src="previewUrl" alt="Selected OCT preview" class="preview" />
            <div v-else class="empty-state">
              <span class="upload-mark">+</span>
              <strong>Select OCT image</strong>
              <span>Drop a file here or browse from your folder.</span>
            </div>
          </label>

          <div class="file-row">
            <div>
              <span class="file-label">Selected file</span>
              <strong>{{ selectedFile?.name || 'None' }}</strong>
            </div>
            <button class="ghost-button" type="button" :disabled="!selectedFile || isLoading" @click="clearFile">
              Clear
            </button>
          </div>

          <button class="primary-button" type="button" :disabled="!canSubmit || isLoading" @click="submit">
            <span v-if="isLoading" class="spinner" aria-hidden="true"></span>
            {{ isLoading ? 'Running inference' : 'Run inference' }}
          </button>
        </section>

        <section class="panel result-panel" aria-live="polite">
          <div class="panel-heading">
            <div>
              <span class="section-kicker">Output</span>
              <h2>Prediction</h2>
            </div>
            <div class="status-line">
              <span
                class="status-dot"
                :class="{ online: backendReady, offline: healthState === 'offline', waking: healthState === 'waking' }"
              ></span>
              <span>{{ healthLabel }}</span>
            </div>
          </div>

          <div v-if="errorMessage" class="message error-message">
            {{ errorMessage }}
          </div>

          <div v-if="result" class="results">
            <div class="metric">
              <span>Disease class</span>
              <strong>{{ result.disease_class }}</strong>
              <small>{{ formatPercent(result.disease_confidence) }} confidence</small>
            </div>

            <div v-if="result.warnings?.length" class="message warning-message">
              <p v-for="warning in result.warnings" :key="warning">{{ warning }}</p>
            </div>
          </div>

          <div v-else-if="!errorMessage" class="waiting">
            <strong>Awaiting input</strong>
            <span>Results will appear here after inference completes.</span>
          </div>

          <div class="result-footer">
            <div>
              <span class="section-kicker">Model inputs</span>
              <strong>Single OCT image tensor</strong>
            </div>
          </div>
        </section>
      </div>
    </section>
  </main>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { pollBackendHealth, predictOctImage } from './api'

const selectedFile = ref(null)
const previewUrl = ref('')
const result = ref(null)
const errorMessage = ref('')
const isLoading = ref(false)
const isDragging = ref(false)
const health = ref(null)
const healthState = ref('checking')
const fileInput = ref(null)
const requestVersion = ref(0)
let healthController = null

const backendReady = computed(() => healthState.value === 'ready' && health.value?.model_loaded === true)
const healthLabel = computed(() => {
  const labels = {
    checking: 'Checking backend',
    waking: 'Waking backend',
    ready: 'Backend ready',
    offline: 'Backend offline'
  }
  return labels[healthState.value]
})
const canSubmit = computed(() => Boolean(selectedFile.value) && backendReady.value)

onMounted(() => startHealthPolling())

onBeforeUnmount(() => {
  healthController?.abort()
  revokePreview()
})

function startHealthPolling() {
  healthController?.abort()
  healthController = new AbortController()
  health.value = null
  void pollBackendHealth({
    signal: healthController.signal,
    onState: (state) => {
      healthState.value = state
    },
    onHealth: (payload) => {
      health.value = payload
    }
  })
}

function handleFileChange(event) {
  const [file] = event.target.files || []
  setFile(file)
}

function handleDrop(event) {
  isDragging.value = false
  const [file] = event.dataTransfer.files || []
  setFile(file)
}

function setFile(file) {
  requestVersion.value += 1
  errorMessage.value = ''
  result.value = null

  if (!file) return
  if (!file.type.startsWith('image/')) {
    selectedFile.value = null
    revokePreview()
    errorMessage.value = 'Please choose an image file.'
    return
  }

  selectedFile.value = file
  revokePreview()
  previewUrl.value = URL.createObjectURL(file)
}

function clearFile() {
  requestVersion.value += 1
  selectedFile.value = null
  result.value = null
  errorMessage.value = ''
  revokePreview()
  if (fileInput.value) fileInput.value.value = ''
}

async function submit() {
  if (!canSubmit.value || isLoading.value) return

  const version = requestVersion.value
  const file = selectedFile.value
  isLoading.value = true
  errorMessage.value = ''
  result.value = null

  try {
    const response = await predictOctImage(file)
    if (version === requestVersion.value && file === selectedFile.value) result.value = response
  } catch (error) {
    if (version === requestVersion.value && file === selectedFile.value) {
      errorMessage.value = error.message
      if (!error.status || error.status === 503) startHealthPolling()
    }
  } finally {
    isLoading.value = false
  }
}

function revokePreview() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = ''
}

function formatPercent(value) {
  if (typeof value !== 'number') return '0%'
  return `${Math.round(value * 100)}%`
}
</script>
