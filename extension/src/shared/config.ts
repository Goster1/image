import type { EdgeBand, Polygon } from './geometry';
import { defaultRectPolygon } from './geometry';
import type { Side } from './messages';

export type Layout = 'L' | 'LR' | 'LRB';

export type DetectionMode = 'hands' | 'motion';

export interface Thresholds {
  /** Hand must stay in a zone at least this long to count as a placement. */
  dwellMs: number;
  /** After a scan, how long to wait for a placement before TIMEOUT. */
  scanTimeoutMs: number;
  /** Hands absent from frame at least this long to count as BEHIND placement. */
  behindAbsenceMs: number;
  /** Mean abs pixel diff (0..255) that counts as motion in a zone. */
  motionThreshold: number;
}

export interface StationConfig {
  name: string;
  cameraDeviceId: string | null;
  layout: Layout;
  zones: {
    LEFT?: Polygon;
    RIGHT?: Polygon;
    BEHIND?: EdgeBand;
  };
  thresholds: Thresholds;
  detectionMode: DetectionMode;
  /** Mirror the preview horizontally (zones stay in image space). */
  mirrorPreview: boolean;
}

export interface AppConfig {
  activeStationId: string;
  stations: Record<string, StationConfig>;
}

export const DEFAULT_THRESHOLDS: Thresholds = {
  dwellMs: 400,
  scanTimeoutMs: 15000,
  behindAbsenceMs: 900,
  motionThreshold: 18,
};

export function layoutSides(layout: Layout): Side[] {
  switch (layout) {
    case 'L':
      return ['LEFT'];
    case 'LR':
      return ['LEFT', 'RIGHT'];
    case 'LRB':
      return ['LEFT', 'RIGHT', 'BEHIND'];
  }
}

export function defaultZonesForLayout(layout: Layout): StationConfig['zones'] {
  const zones: StationConfig['zones'] = {
    LEFT: defaultRectPolygon(0.02, 0.25, 0.28, 0.6),
  };
  if (layout === 'LR' || layout === 'LRB') {
    zones.RIGHT = defaultRectPolygon(0.7, 0.25, 0.28, 0.6);
  }
  if (layout === 'LRB') {
    zones.BEHIND = { edge: 'bottom', thickness: 0.15 };
  }
  return zones;
}

export function defaultStation(name = 'Stanica 1'): StationConfig {
  return {
    name,
    cameraDeviceId: null,
    layout: 'LR',
    zones: defaultZonesForLayout('LR'),
    thresholds: { ...DEFAULT_THRESHOLDS },
    detectionMode: 'hands',
    mirrorPreview: false,
  };
}

export function defaultAppConfig(): AppConfig {
  return { activeStationId: 'station-1', stations: { 'station-1': defaultStation() } };
}

const STORAGE_KEY = 'vk-config';

export async function loadConfig(): Promise<AppConfig> {
  const raw = await chrome.storage.local.get(STORAGE_KEY);
  const cfg = raw[STORAGE_KEY] as AppConfig | undefined;
  if (!cfg || !cfg.stations || !cfg.stations[cfg.activeStationId]) {
    return defaultAppConfig();
  }
  // Fill in any thresholds added after the config was first saved.
  for (const st of Object.values(cfg.stations)) {
    st.thresholds = { ...DEFAULT_THRESHOLDS, ...st.thresholds };
  }
  return cfg;
}

export async function saveConfig(cfg: AppConfig): Promise<void> {
  await chrome.storage.local.set({ [STORAGE_KEY]: cfg });
}
