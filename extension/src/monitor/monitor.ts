/**
 * Monitor page orchestrator: camera → hand tracking / motion detection →
 * verifier → results (runtime messages, log, UI). Also hosts the
 * calibration UI and the violations log.
 */
import type { AppConfig, StationConfig } from '../shared/config';
import {
  defaultStation,
  defaultZonesForLayout,
  layoutSides,
  loadConfig,
  saveConfig,
} from '../shared/config';
import type { Point } from '../shared/geometry';
import { pointInEdgeBand, pointInPolygon } from '../shared/geometry';
import type { RuntimeMessage, Side, VerificationResult } from '../shared/messages';
import { isScanEvent, SIDE_LABELS_SK } from '../shared/messages';
import type { CameraSource } from './camera';
import { listCameras, openCamera, openVideoFile } from './camera';
import { CalibrationEditor } from './calibration';
import { HandTracker } from './hand-tracker';
import { MotionDetector } from './motion-fallback';
import { appendLog, clearLog, downloadCsv, readLog } from './log';
import type { HandFrame } from './verifier';
import { DEFAULT_VERIFIER_OPTIONS, Verifier } from './verifier';

// --- DOM ---
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const video = $<HTMLVideoElement>('video');
const overlay = $<HTMLCanvasElement>('overlay');
const statusPill = $('status-pill');
const scanState = $('scan-state');
const cameraSelect = $<HTMLSelectElement>('camera-select');
const recentResults = $('recent-results');
const stationSelect = $<HTMLSelectElement>('station-select');
const behindEdgeSelect = $<HTMLSelectElement>('behind-edge');
const detectionModeSelect = $<HTMLSelectElement>('detection-mode');
const dwellInput = $<HTMLInputElement>('dwell-ms');
const timeoutInput = $<HTMLInputElement>('timeout-ms');
const editZonesCheckbox = $<HTMLInputElement>('edit-zones');
const mirrorCheckbox = $<HTMLInputElement>('mirror-preview');

// --- state ---
let config: AppConfig;
let station: StationConfig;
let source: CameraSource | null = null;
const tracker = new HandTracker();
let trackerFailed = false;
const motion = new MotionDetector();
let verifier: Verifier;
let editor: CalibrationEditor;
/** 3-frame zone history per hand id, for majority smoothing. */
const zoneHistory = new Map<string, ('LEFT' | 'RIGHT' | null)[]>();
let lastMotionTs = 0;
let lastMotionEnergy: Record<string, number> = {};

function activeStation(): StationConfig {
  return config.stations[config.activeStationId];
}

function verifierOptions() {
  return {
    ...DEFAULT_VERIFIER_OPTIONS,
    dwellMs: station.thresholds.dwellMs,
    scanTimeoutMs: station.thresholds.scanTimeoutMs,
    behindAbsenceMs: station.thresholds.behindAbsenceMs,
    enabledSides: layoutSides(station.layout),
  };
}

async function persist() {
  await saveConfig(config);
  const msg: RuntimeMessage = { type: 'CONFIG_UPDATED', stationId: config.activeStationId };
  void chrome.runtime.sendMessage(msg).catch(() => {});
}

function applyStationToUi() {
  for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="layout"]')) {
    radio.checked = radio.value === station.layout;
  }
  behindEdgeSelect.value = station.zones.BEHIND?.edge ?? 'bottom';
  detectionModeSelect.value = station.detectionMode;
  dwellInput.value = String(station.thresholds.dwellMs);
  timeoutInput.value = String(station.thresholds.scanTimeoutMs);
  mirrorCheckbox.checked = station.mirrorPreview;
  video.classList.toggle('mirrored', station.mirrorPreview);
  overlay.classList.toggle('mirrored', station.mirrorPreview);
  editor.setZones(station.zones);
  editor.setMirrored(station.mirrorPreview);
  const motionZones: Parameters<typeof motion.setZones>[0] = {};
  if (station.zones.LEFT) motionZones['LEFT'] = station.zones.LEFT;
  if (station.zones.RIGHT) motionZones['RIGHT'] = station.zones.RIGHT;
  if (station.zones.BEHIND) motionZones['BEHIND'] = station.zones.BEHIND;
  motion.setZones(motionZones);
  verifier.setOptions(verifierOptions());
}

function rebuildStationSelect() {
  stationSelect.innerHTML = '';
  for (const [id, st] of Object.entries(config.stations)) {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = st.name;
    opt.selected = id === config.activeStationId;
    stationSelect.appendChild(opt);
  }
}

// --- results UI ---
const OUTCOME_LABELS: Record<string, string> = {
  MATCH: '✓ správne',
  MISMATCH: '✗ zlá strana',
  TIMEOUT: '⌛ nepoložené',
  SUPERSEDED: '↷ preskočené',
};

function showScanState(text: string, cls: string) {
  scanState.hidden = false;
  scanState.className = `scan-state ${cls}`;
  scanState.textContent = text;
}

async function handleResult(result: VerificationResult) {
  void chrome.runtime.sendMessage(result).catch(() => {});
  await appendLog(result);

  const li = document.createElement('li');
  li.className = result.outcome;
  const detected = result.detectedSide ? ` → ${SIDE_LABELS_SK[result.detectedSide]}` : '';
  li.innerHTML = `<span>${result.sku}</span><span>${OUTCOME_LABELS[result.outcome]}${detected}</span>`;
  recentResults.prepend(li);
  while (recentResults.children.length > 20) recentResults.lastChild?.remove();

  if (result.outcome === 'MATCH') {
    showScanState(`✓ ${result.sku} — správne`, 'match');
  } else if (result.outcome === 'MISMATCH') {
    showScanState(
      `✗ ${result.sku} — položené ${result.detectedSide ? SIDE_LABELS_SK[result.detectedSide] : '?'}, malo byť ${SIDE_LABELS_SK[result.expectedSide]}`,
      'mismatch',
    );
  } else if (result.outcome === 'TIMEOUT') {
    showScanState(`⌛ ${result.sku} — položenie nezaznamenané`, 'waiting');
  }
  window.setTimeout(() => {
    if (!verifier.activeScanId) scanState.hidden = true;
  }, 4000);
}

// --- pipeline ---
function zoneOfPoint(p: Point): 'LEFT' | 'RIGHT' | null {
  if (station.zones.LEFT && pointInPolygon(p, station.zones.LEFT)) return 'LEFT';
  if (station.zones.RIGHT && pointInPolygon(p, station.zones.RIGHT)) return 'RIGHT';
  return null;
}

function smoothZone(id: string, zone: 'LEFT' | 'RIGHT' | null): 'LEFT' | 'RIGHT' | null {
  let hist = zoneHistory.get(id);
  if (!hist) {
    hist = [];
    zoneHistory.set(id, hist);
  }
  hist.push(zone);
  if (hist.length > 3) hist.shift();
  const counts = new Map<'LEFT' | 'RIGHT' | null, number>();
  for (const z of hist) counts.set(z, (counts.get(z) ?? 0) + 1);
  let best: 'LEFT' | 'RIGHT' | null = zone;
  let bestCount = 0;
  for (const [z, c] of counts) {
    if (c > bestCount) {
      best = z;
      bestCount = c;
    }
  }
  return best;
}

function processFrame(ts: number) {
  if (!source) return;

  // motion sampling at ~10 fps (also drives the BEHIND corroboration)
  if (ts - lastMotionTs >= 100) {
    lastMotionTs = ts;
    lastMotionEnergy = motion.sample(video, ts).zoneEnergy;
  }
  const behindMotion = (lastMotionEnergy['BEHIND'] ?? 0) > station.thresholds.motionThreshold;

  const frame: HandFrame = { ts, hands: [], behindMotion };
  const dots: Point[] = [];

  if (station.detectionMode === 'hands' && tracker.ready) {
    const obs = tracker.detect(video, ts);
    for (const hand of obs.hands) {
      dots.push(hand.palm);
      frame.hands.push({
        id: hand.id,
        zone: smoothZone(hand.id, zoneOfPoint(hand.palm)),
        inBehindBand: station.zones.BEHIND
          ? pointInEdgeBand(hand.palm, station.zones.BEHIND)
          : false,
      });
    }
  } else {
    // Motion-only fallback: a zone with live motion acts as a pseudo-hand,
    // so the same dwell/retract state machine applies.
    for (const side of ['LEFT', 'RIGHT'] as const) {
      if ((lastMotionEnergy[side] ?? 0) > station.thresholds.motionThreshold) {
        frame.hands.push({ id: `motion-${side}`, zone: side, inBehindBand: false });
      }
    }
    if (behindMotion) {
      frame.hands.push({ id: 'motion-BEHIND', zone: null, inBehindBand: true });
    }
  }

  const activeZones = new Set<Side>();
  for (const h of frame.hands) {
    if (h.zone) activeZones.add(h.zone);
    if (h.inBehindBand) activeZones.add('BEHIND');
  }

  for (const result of verifier.onFrame(frame)) {
    void handleResult(result);
  }

  editor.setHandDots(dots);
  editor.render(activeZones);
}

function startLoop() {
  const useRvfc = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
  let lastTs = 0;
  const step = () => {
    const now = performance.now();
    // throttle inference to ~15 fps
    if (now - lastTs >= 66) {
      lastTs = now;
      try {
        processFrame(now);
      } catch (err) {
        console.error('frame processing failed', err);
      }
    }
    if (useRvfc && source) {
      video.requestVideoFrameCallback(() => step());
    } else {
      requestAnimationFrame(step);
    }
  };
  step();
}

// --- camera controls ---
function setCameraStatus(ok: boolean, label: string) {
  statusPill.className = `pill ${ok ? 'pill-on' : 'pill-off'}`;
  statusPill.textContent = label;
}

async function startCamera() {
  source?.stop();
  motion.reset();
  try {
    source = await openCamera(video, cameraSelect.value || station.cameraDeviceId);
    station.cameraDeviceId = cameraSelect.value || null;
    await persist();
    setCameraStatus(true, trackerFailed ? 'Kamera beží (len pohyb)' : 'Kamera beží');
    syncOverlaySize();
    // Labels are only available after permission was granted once.
    await populateCameras();
  } catch (err) {
    console.error(err);
    setCameraStatus(false, 'Kamera zlyhala');
  }
}

async function populateCameras() {
  const cams = await listCameras();
  const current = cameraSelect.value;
  cameraSelect.innerHTML = '';
  cams.forEach((cam, i) => {
    const opt = document.createElement('option');
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Kamera ${i + 1}`;
    cameraSelect.appendChild(opt);
  });
  cameraSelect.value = current || station.cameraDeviceId || (cams[0]?.deviceId ?? '');
}

function syncOverlaySize() {
  overlay.width = video.videoWidth || 1280;
  overlay.height = video.videoHeight || 720;
}
video.addEventListener('loadedmetadata', syncOverlaySize);

// --- wiring ---
async function main() {
  config = await loadConfig();
  station = activeStation();
  verifier = new Verifier(verifierOptions());
  editor = new CalibrationEditor(overlay, station.zones, () => {
    void persist();
  });

  rebuildStationSelect();
  applyStationToUi();
  await populateCameras();

  // tabs
  for (const tab of document.querySelectorAll<HTMLButtonElement>('.tab')) {
    tab.addEventListener('click', () => {
      for (const t of document.querySelectorAll('.tab')) t.classList.remove('active');
      tab.classList.add('active');
      for (const p of document.querySelectorAll<HTMLElement>('.tab-panel')) {
        p.hidden = p.dataset.panel !== tab.dataset.tab;
      }
      const isCalib = tab.dataset.tab === 'calibration';
      editor.setEditable(isCalib && editZonesCheckbox.checked);
      if (tab.dataset.tab === 'log') void renderLog();
    });
  }

  $('start-camera').addEventListener('click', () => void startCamera());
  $<HTMLInputElement>('video-file').addEventListener('change', async (ev) => {
    const file = (ev.target as HTMLInputElement).files?.[0];
    if (!file) return;
    source?.stop();
    motion.reset();
    source = await openVideoFile(video, file);
    setCameraStatus(true, 'Testovacie video');
    syncOverlaySize();
  });

  // station management
  stationSelect.addEventListener('change', async () => {
    config.activeStationId = stationSelect.value;
    station = activeStation();
    applyStationToUi();
    await persist();
  });
  $('station-add').addEventListener('click', async () => {
    const name = prompt('Názov novej stanice:', `Stanica ${Object.keys(config.stations).length + 1}`);
    if (!name) return;
    const id = `station-${crypto.randomUUID().slice(0, 8)}`;
    config.stations[id] = defaultStation(name);
    config.activeStationId = id;
    station = activeStation();
    rebuildStationSelect();
    applyStationToUi();
    await persist();
  });
  $('station-rename').addEventListener('click', async () => {
    const name = prompt('Nový názov stanice:', station.name);
    if (!name) return;
    station.name = name;
    rebuildStationSelect();
    await persist();
  });

  // layout + zones
  for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="layout"]')) {
    radio.addEventListener('change', async () => {
      if (!radio.checked) return;
      station.layout = radio.value as StationConfig['layout'];
      station.zones = defaultZonesForLayout(station.layout);
      applyStationToUi();
      await persist();
    });
  }
  behindEdgeSelect.addEventListener('change', async () => {
    if (station.zones.BEHIND) {
      station.zones.BEHIND.edge = behindEdgeSelect.value as 'left' | 'right' | 'top' | 'bottom';
      await persist();
    }
  });
  editZonesCheckbox.addEventListener('change', () => {
    editor.setEditable(editZonesCheckbox.checked);
  });
  $('reset-zones').addEventListener('click', async () => {
    station.zones = defaultZonesForLayout(station.layout);
    applyStationToUi();
    await persist();
  });

  // detection settings
  mirrorCheckbox.addEventListener('change', async () => {
    station.mirrorPreview = mirrorCheckbox.checked;
    applyStationToUi();
    await persist();
  });
  detectionModeSelect.addEventListener('change', async () => {
    station.detectionMode = detectionModeSelect.value as StationConfig['detectionMode'];
    await persist();
  });
  dwellInput.addEventListener('change', async () => {
    station.thresholds.dwellMs = Number(dwellInput.value) || 400;
    verifier.setOptions(verifierOptions());
    await persist();
  });
  timeoutInput.addEventListener('change', async () => {
    station.thresholds.scanTimeoutMs = Number(timeoutInput.value) || 15000;
    verifier.setOptions(verifierOptions());
    await persist();
  });

  // log tab
  $('export-csv').addEventListener('click', async () => downloadCsv(await readLog()));
  $('clear-log').addEventListener('click', async () => {
    await clearLog();
    await renderLog();
  });

  // runtime messages
  chrome.runtime.onMessage.addListener((message: unknown) => {
    if (isScanEvent(message)) {
      for (const result of verifier.onScan(message)) void handleResult(result);
      showScanState(
        `Čakám: ${message.sku} → ${SIDE_LABELS_SK[message.expectedSide]}`,
        'waiting',
      );
    }
  });

  // heartbeat
  window.setInterval(() => {
    const msg: RuntimeMessage = {
      type: 'MONITOR_HEARTBEAT',
      cameraOk: source !== null,
      ts: Date.now(),
    };
    void chrome.runtime.sendMessage(msg).catch(() => {});
  }, 2000);

  // MediaPipe init (highest-risk piece — fall back to motion mode on failure)
  try {
    await tracker.init();
  } catch (err) {
    console.error('MediaPipe init failed, falling back to motion detection', err);
    trackerFailed = true;
    station.detectionMode = 'motion';
    detectionModeSelect.value = 'motion';
  }

  startLoop();
}

async function renderLog() {
  const entries = await readLog();
  const tbody = document.querySelector('#log-table tbody')!;
  tbody.innerHTML = '';
  for (const e of entries.slice().reverse()) {
    const tr = document.createElement('tr');
    tr.className = e.outcome;
    tr.innerHTML = `<td>${new Date(e.ts).toLocaleString('sk')}</td><td>${e.sku}</td><td>${e.expected}</td><td>${e.detected ?? '—'}</td><td>${OUTCOME_LABELS[e.outcome] ?? e.outcome}</td>`;
    tbody.appendChild(tr);
  }
}

void main();
