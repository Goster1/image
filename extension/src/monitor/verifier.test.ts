import { describe, expect, it } from 'vitest';
import type { ScanEvent } from '../shared/messages';
import type { HandFrame } from './verifier';
import { DEFAULT_VERIFIER_OPTIONS, Verifier } from './verifier';

const OPTS = {
  ...DEFAULT_VERIFIER_OPTIONS,
  enabledSides: ['LEFT', 'RIGHT', 'BEHIND'] as ('LEFT' | 'RIGHT' | 'BEHIND')[],
};

function scan(expectedSide: 'LEFT' | 'RIGHT' | 'BEHIND', ts = 0): ScanEvent {
  return { type: 'SCAN_EVENT', scanId: `scan-${ts}`, sku: 'SKU1', expectedSide, ts };
}

function frame(
  ts: number,
  hands: HandFrame['hands'] = [],
  behindMotion = false,
): HandFrame {
  return { ts, hands, behindMotion };
}

const inZone = (zone: 'LEFT' | 'RIGHT' | null, id = 'Right-0') => ({
  id,
  zone,
  inBehindBand: false,
});

/** Feed frames every 66 ms from `from` to `to` with the given hands. */
function feed(
  v: Verifier,
  from: number,
  to: number,
  hands: HandFrame['hands'],
  behindMotion = false,
) {
  const results = [];
  for (let ts = from; ts <= to; ts += 66) {
    results.push(...v.onFrame(frame(ts, hands, behindMotion)));
  }
  return results;
}

describe('Verifier', () => {
  it('matches when the hand dwells in the expected zone and retracts', () => {
    const v = new Verifier(OPTS);
    expect(v.onScan(scan('LEFT'))).toEqual([]);
    let results = feed(v, 100, 700, [inZone('LEFT')]);
    expect(results).toEqual([]);
    results = feed(v, 766, 1000, [inZone(null)]);
    expect(results).toHaveLength(1);
    expect(results[0].outcome).toBe('MATCH');
    expect(results[0].detectedSide).toBe('LEFT');
  });

  it('flags a mismatch when placed on the wrong side', () => {
    const v = new Verifier(OPTS);
    v.onScan(scan('LEFT'));
    feed(v, 100, 700, [inZone('RIGHT')]);
    const results = feed(v, 766, 2500, [inZone(null)]);
    expect(results).toHaveLength(1);
    expect(results[0].outcome).toBe('MISMATCH');
    expect(results[0].detectedSide).toBe('RIGHT');
    expect(results[0].expectedSide).toBe('LEFT');
  });

  it('ignores a hand briefly passing through a zone', () => {
    const v = new Verifier(OPTS);
    v.onScan(scan('RIGHT'));
    // pass through LEFT for only ~130 ms (< dwellMs)
    feed(v, 100, 230, [inZone('LEFT')]);
    feed(v, 296, 350, [inZone(null)]);
    // then place properly on RIGHT
    feed(v, 400, 1000, [inZone('RIGHT')]);
    const results = feed(v, 1066, 1300, [inZone(null)]);
    expect(results).toHaveLength(1);
    expect(results[0].outcome).toBe('MATCH');
    expect(results[0].detectedSide).toBe('RIGHT');
  });

  it('judges the final zone when the worker corrects themselves', () => {
    const v = new Verifier(OPTS);
    v.onScan(scan('RIGHT'));
    // first dwells in LEFT (wrong), then moves to RIGHT before settling
    feed(v, 100, 700, [inZone('LEFT')]);
    const corrected = feed(v, 766, 1400, [inZone('RIGHT')]);
    // moving to the right zone should not have fired a mismatch yet
    expect(corrected.filter((r) => r.outcome === 'MISMATCH')).toEqual([]);
    const results = feed(v, 1466, 1700, [inZone(null)]);
    expect(results).toHaveLength(1);
    expect(results[0].outcome).toBe('MATCH');
  });

  it('times out when nothing is placed', () => {
    const v = new Verifier(OPTS);
    v.onScan(scan('LEFT'));
    const results = feed(v, 100, 16000, []);
    expect(results).toHaveLength(1);
    expect(results[0].outcome).toBe('TIMEOUT');
    expect(results[0].detectedSide).toBeNull();
  });

  it('supersedes the previous scan when a new one arrives', () => {
    const v = new Verifier(OPTS);
    v.onScan(scan('LEFT', 0));
    const results = v.onScan(scan('RIGHT', 5000));
    expect(results).toHaveLength(1);
    expect(results[0].outcome).toBe('SUPERSEDED');
    expect(results[0].scanId).toBe('scan-0');
    expect(v.activeScanId).toBe('scan-5000');
  });

  it('detects BEHIND via edge band + hand absence + motion', () => {
    const v = new Verifier(OPTS);
    v.onScan(scan('BEHIND'));
    // hand moves into the behind band with motion, then leaves the frame
    feed(v, 100, 400, [{ id: 'Right-0', zone: null, inBehindBand: true }], true);
    const results = feed(v, 466, 1600, []);
    const match = results.find((r) => r.outcome === 'MATCH');
    expect(match).toBeDefined();
    expect(match!.detectedSide).toBe('BEHIND');
  });

  it('does not fire BEHIND when hands simply leave the frame without the band', () => {
    const v = new Verifier(OPTS);
    v.onScan(scan('BEHIND'));
    feed(v, 100, 400, [inZone(null)]);
    const results = feed(v, 466, 3000, []);
    expect(results.find((r) => r.outcome === 'MATCH')).toBeUndefined();
  });

  it('matches on long dwell without retraction', () => {
    const v = new Verifier(OPTS);
    v.onScan(scan('LEFT'));
    // hand stays in the correct zone for 4× dwellMs without leaving
    const results = feed(v, 100, 2000, [inZone('LEFT')]);
    const match = results.find((r) => r.outcome === 'MATCH');
    expect(match).toBeDefined();
  });

  it('treats a hand vanishing inside a zone as retraction', () => {
    const v = new Verifier(OPTS);
    v.onScan(scan('LEFT'));
    feed(v, 100, 700, [inZone('LEFT')]);
    // hand disappears (occluded behind the shelf) instead of visibly leaving
    const results = feed(v, 766, 1600, []);
    expect(results.find((r) => r.outcome === 'MATCH')).toBeDefined();
  });
});
