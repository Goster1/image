/**
 * Demo picking page. Simulates the WMS app the worker uses:
 *  - captures keyboard-wedge scanner input (characters + Enter),
 *  - assigns the target side for the scanned SKU,
 *  - shows a big directional card,
 *  - dispatches CustomEvent('vk-scan') for the extension content script,
 *  - renders outcome history from CustomEvent('vk-result').
 */

type Side = 'LEFT' | 'RIGHT' | 'BEHIND';
type Layout = 'L' | 'LR' | 'LRB';

const SIDE_LABELS: Record<Side, string> = {
  LEFT: 'VĽAVO',
  RIGHT: 'VPRAVO',
  BEHIND: 'ZA SEBA',
};
const SIDE_ARROWS: Record<Side, string> = { LEFT: '←', RIGHT: '→', BEHIND: '↶' };
const SIDE_CLASSES: Record<Side, string> = { LEFT: 'left', RIGHT: 'right', BEHIND: 'behind' };
const OUTCOME_LABELS: Record<string, string> = {
  MATCH: '✓ správne',
  MISMATCH: '✗ ZLÁ STRANA',
  TIMEOUT: '⌛ nepoložené',
  SUPERSEDED: '↷ preskočené',
};

const card = document.getElementById('direction-card')!;
const arrowEl = document.getElementById('arrow')!;
const sideEl = document.getElementById('side-label')!;
const skuEl = document.getElementById('sku-label')!;
const historyEl = document.getElementById('history')!;
const skuInput = document.getElementById('sku-input') as HTMLInputElement;
const scanBtn = document.getElementById('scan-btn') as HTMLButtonElement;
const layoutSelect = document.getElementById('layout-select') as HTMLSelectElement;

function enabledSides(): Side[] {
  const layout = layoutSelect.value as Layout;
  if (layout === 'L') return ['LEFT'];
  if (layout === 'LR') return ['LEFT', 'RIGHT'];
  return ['LEFT', 'RIGHT', 'BEHIND'];
}

/** Deterministic side per SKU so re-scanning the same code gives the same side. */
function sideForSku(sku: string): Side {
  const sides = enabledSides();
  let h = 0;
  for (const ch of sku) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return sides[h % sides.length];
}

const pendingScans = new Map<string, { sku: string; side: Side }>();

function emitScan(sku: string, forcedSide?: Side) {
  const side = forcedSide ?? sideForSku(sku);
  const scanId = crypto.randomUUID();
  pendingScans.set(scanId, { sku, side });

  card.className = SIDE_CLASSES[side];
  arrowEl.textContent = SIDE_ARROWS[side];
  sideEl.textContent = SIDE_LABELS[side];
  skuEl.textContent = sku;

  window.dispatchEvent(
    new CustomEvent('vk-scan', { detail: { scanId, sku, expectedSide: side } }),
  );
}

// --- keyboard wedge: scanners type the code and finish with Enter ---
let wedgeBuffer = '';
let wedgeTimer: number | undefined;

document.addEventListener('keydown', (ev) => {
  if (ev.target === skuInput) return; // manual field handles its own Enter

  if (ev.key === 'F1' || ev.key === 'F2' || ev.key === 'F3') {
    ev.preventDefault();
    const forced: Side = ev.key === 'F1' ? 'LEFT' : ev.key === 'F2' ? 'RIGHT' : 'BEHIND';
    if (!enabledSides().includes(forced)) return;
    emitScan(`TEST-${forced}`, forced);
    return;
  }

  if (ev.key === 'Enter') {
    if (wedgeBuffer.length >= 3) emitScan(wedgeBuffer);
    wedgeBuffer = '';
    return;
  }
  if (ev.key.length === 1) {
    wedgeBuffer += ev.key;
    clearTimeout(wedgeTimer);
    // Human typing is slow; a scanner sends the whole code in <100 ms.
    wedgeTimer = window.setTimeout(() => (wedgeBuffer = ''), 300);
  }
});

function manualScan() {
  const sku = skuInput.value.trim();
  if (sku.length < 1) return;
  emitScan(sku);
  skuInput.value = '';
}

scanBtn.addEventListener('click', manualScan);
skuInput.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') manualScan();
});

// --- outcome history from the extension ---
window.addEventListener('vk-result', (ev) => {
  const r = (ev as CustomEvent).detail as {
    scanId: string;
    sku: string;
    outcome: string;
    expectedSide: Side;
    detectedSide: Side | null;
  };
  pendingScans.delete(r.scanId);
  const li = document.createElement('li');
  li.className = r.outcome;
  const detected = r.detectedSide ? ` (položené ${SIDE_LABELS[r.detectedSide]})` : '';
  li.innerHTML = `<span>${r.sku}</span><span>${OUTCOME_LABELS[r.outcome] ?? r.outcome}${detected}</span>`;
  historyEl.prepend(li);
  while (historyEl.children.length > 30) historyEl.lastChild?.remove();
});
