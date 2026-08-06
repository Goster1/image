/**
 * Content script for the picking page (demo WMS or a real one later).
 *
 * Integration contract with the page:
 *  - page dispatches `window CustomEvent('vk-scan', { detail: { scanId, sku, expectedSide } })`
 *  - this script dispatches `CustomEvent('vk-result', { detail: VerificationResult })` back
 *
 * On MISMATCH it renders a full-width red banner with sound; if the monitor
 * page is not running when a scan happens, it shows a yellow warning instead.
 */
import type { MonitorStatusReply, ScanEvent, Side, VerificationResult } from '../shared/messages';
import { isVerificationResult, SIDE_LABELS_SK } from '../shared/messages';

const BANNER_ID = 'vk-banner';
let bannerTimer: number | undefined;

function showBanner(kind: 'mismatch' | 'warning', text: string) {
  removeBanner();
  const el = document.createElement('div');
  el.id = BANNER_ID;
  el.className = `vk-banner vk-banner--${kind}`;
  el.textContent = text;
  document.documentElement.appendChild(el);
  bannerTimer = window.setTimeout(removeBanner, 5000);
}

function removeBanner() {
  if (bannerTimer !== undefined) {
    clearTimeout(bannerTimer);
    bannerTimer = undefined;
  }
  document.getElementById(BANNER_ID)?.remove();
}

function playAlert() {
  const audio = new Audio(chrome.runtime.getURL('sounds/alert.wav'));
  // The scan itself was a user gesture (keyboard input), so playback is allowed.
  void audio.play().catch(() => {});
}

window.addEventListener('vk-scan', (ev) => {
  const detail = (ev as CustomEvent).detail as
    | { scanId?: string; sku?: string; expectedSide?: Side }
    | undefined;
  if (!detail?.scanId || !detail.sku || !detail.expectedSide) return;

  removeBanner(); // new scan clears the previous alert

  const msg: ScanEvent = {
    type: 'SCAN_EVENT',
    scanId: detail.scanId,
    sku: detail.sku,
    expectedSide: detail.expectedSide,
    ts: Date.now(),
  };
  void chrome.runtime.sendMessage(msg).catch(() => {});

  // Warn the worker when nobody is actually watching the camera.
  void chrome.runtime
    .sendMessage({ type: 'MONITOR_STATUS_QUERY' })
    .then((reply: MonitorStatusReply) => {
      if (!reply?.monitorAlive || !reply.cameraOk) {
        showBanner('warning', 'Kamera nebeží — kontrola položenia nie je aktívna!');
      }
    })
    .catch(() => {});
});

chrome.runtime.onMessage.addListener((message: unknown) => {
  if (!isVerificationResult(message)) return;
  const result = message as VerificationResult;

  // Let the page render its own history of outcomes.
  window.dispatchEvent(new CustomEvent('vk-result', { detail: result }));

  if (result.outcome === 'MISMATCH') {
    showBanner(
      'mismatch',
      `ZLÁ STRANA! Polož na: ${SIDE_LABELS_SK[result.expectedSide]}`,
    );
    playAlert();
  }
});
