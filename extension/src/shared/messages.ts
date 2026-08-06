/**
 * Message protocol shared by the demo/WMS page, content script,
 * service worker and the monitor page.
 *
 * Flow:
 *   picking page --CustomEvent('vk-scan')--> content script
 *   content script --SCAN_EVENT--> service worker --> monitor page
 *   monitor page --VERIFICATION_RESULT--> service worker --> content script
 *   content script --CustomEvent('vk-result')--> picking page
 */

export type Side = 'LEFT' | 'RIGHT' | 'BEHIND';

export type Outcome = 'MATCH' | 'MISMATCH' | 'TIMEOUT' | 'SUPERSEDED';

export interface ScanEvent {
  type: 'SCAN_EVENT';
  scanId: string;
  sku: string;
  expectedSide: Side;
  ts: number;
}

export interface VerificationResult {
  type: 'VERIFICATION_RESULT';
  scanId: string;
  sku: string;
  outcome: Outcome;
  expectedSide: Side;
  detectedSide: Side | null;
  /** 0..1 heuristic confidence of the detection */
  confidence: number;
  ts: number;
}

export interface MonitorHeartbeat {
  type: 'MONITOR_HEARTBEAT';
  cameraOk: boolean;
  ts: number;
}

/** Content script asks whether the monitor is alive (SW answers from last heartbeat). */
export interface MonitorStatusQuery {
  type: 'MONITOR_STATUS_QUERY';
}

export interface MonitorStatusReply {
  type: 'MONITOR_STATUS_REPLY';
  monitorAlive: boolean;
  cameraOk: boolean;
}

export interface ConfigUpdated {
  type: 'CONFIG_UPDATED';
  stationId: string;
}

export type RuntimeMessage =
  | ScanEvent
  | VerificationResult
  | MonitorHeartbeat
  | MonitorStatusQuery
  | ConfigUpdated;

export const SIDE_LABELS_SK: Record<Side, string> = {
  LEFT: 'VĽAVO',
  RIGHT: 'VPRAVO',
  BEHIND: 'ZA SEBA',
};

export const SIDE_ARROWS: Record<Side, string> = {
  LEFT: '←',
  RIGHT: '→',
  BEHIND: '↶',
};

export function isScanEvent(m: unknown): m is ScanEvent {
  const x = m as ScanEvent;
  return (
    !!x &&
    x.type === 'SCAN_EVENT' &&
    typeof x.scanId === 'string' &&
    typeof x.sku === 'string' &&
    (x.expectedSide === 'LEFT' || x.expectedSide === 'RIGHT' || x.expectedSide === 'BEHIND')
  );
}

export function isVerificationResult(m: unknown): m is VerificationResult {
  const x = m as VerificationResult;
  return !!x && x.type === 'VERIFICATION_RESULT' && typeof x.scanId === 'string';
}
