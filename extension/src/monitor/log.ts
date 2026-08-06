/**
 * Violations/results log: capped ring buffer in chrome.storage.local
 * plus CSV export.
 */
import type { VerificationResult } from '../shared/messages';

const LOG_KEY = 'vk-log';
const MAX_ENTRIES = 1000;

export interface LogEntry {
  ts: number;
  scanId: string;
  sku: string;
  expected: string;
  detected: string | null;
  outcome: string;
}

export async function appendLog(result: VerificationResult): Promise<void> {
  const entry: LogEntry = {
    ts: result.ts,
    scanId: result.scanId,
    sku: result.sku,
    expected: result.expectedSide,
    detected: result.detectedSide,
    outcome: result.outcome,
  };
  const entries = await readLog();
  entries.push(entry);
  while (entries.length > MAX_ENTRIES) entries.shift();
  await chrome.storage.local.set({ [LOG_KEY]: entries });
}

export async function readLog(): Promise<LogEntry[]> {
  const raw = await chrome.storage.local.get(LOG_KEY);
  return (raw[LOG_KEY] as LogEntry[] | undefined) ?? [];
}

export async function clearLog(): Promise<void> {
  await chrome.storage.local.remove(LOG_KEY);
}

export function toCsv(entries: LogEntry[]): string {
  const header = 'timestamp,scanId,sku,expected,detected,outcome';
  const rows = entries.map((e) =>
    [new Date(e.ts).toISOString(), e.scanId, e.sku, e.expected, e.detected ?? '', e.outcome]
      .map((v) => `"${String(v).replaceAll('"', '""')}"`)
      .join(','),
  );
  return [header, ...rows].join('\n');
}

export function downloadCsv(entries: LogEntry[]): void {
  const blob = new Blob([toCsv(entries)], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `video-kontrola-log-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
