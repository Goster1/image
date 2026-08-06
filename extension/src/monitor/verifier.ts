/**
 * Placement verification state machine. Pure logic — no DOM, no chrome.* —
 * so it is unit-testable with scripted observation sequences.
 *
 * Per scan: IDLE → AWAITING_PLACEMENT → MATCH | MISMATCH | TIMEOUT | SUPERSEDED → IDLE
 *
 * A "placement candidate" in zone Z fires when a hand stays in Z at least
 * `dwellMs` and then leaves Z (dwell + retraction filters a hand merely
 * passing through a zone). The last candidate before hands settle back to
 * neutral wins, so a worker who corrects themselves mid-air is judged by
 * the final zone. BEHIND is a heuristic: the last tracked hand position was
 * inside the configured edge band and then all hands left the frame for at
 * least `behindAbsenceMs`.
 */
import type { Outcome, ScanEvent, Side, VerificationResult } from '../shared/messages';

export interface HandFrame {
  ts: number;
  /** One entry per visible hand. `zone` is the polygon zone the hand is in. */
  hands: { id: string; zone: 'LEFT' | 'RIGHT' | null; inBehindBand: boolean }[];
  /** True when the BEHIND edge band currently shows motion energy. */
  behindMotion: boolean;
}

export interface VerifierOptions {
  dwellMs: number;
  scanTimeoutMs: number;
  behindAbsenceMs: number;
  /** How long hands must stay out of all zones before a wrong candidate is final. */
  retractGraceMs: number;
  enabledSides: Side[];
}

export const DEFAULT_VERIFIER_OPTIONS: Omit<VerifierOptions, 'enabledSides'> = {
  dwellMs: 400,
  scanTimeoutMs: 15000,
  behindAbsenceMs: 900,
  retractGraceMs: 600,
};

interface ActiveScan {
  scanId: string;
  sku: string;
  expectedSide: Side;
  deadline: number;
  /** ts + zone of the best (latest) placement candidate so far */
  candidate: { side: Side; ts: number } | null;
  candidateCount: number;
}

interface HandTrack {
  zone: 'LEFT' | 'RIGHT' | null;
  enteredAt: number;
}

export class Verifier {
  private opts: VerifierOptions;
  private scan: ActiveScan | null = null;
  private tracks = new Map<string, HandTrack>();
  private lastHandsPresentTs = 0;
  private lastHandInBandTs = -Infinity;
  private behindMotionSeenTs = -Infinity;
  private lastNoZoneTs = 0;

  constructor(opts: VerifierOptions) {
    this.opts = opts;
  }

  setOptions(opts: VerifierOptions) {
    this.opts = opts;
  }

  get activeScanId(): string | null {
    return this.scan?.scanId ?? null;
  }

  /** A new scan arrived. May finalize the previous scan as SUPERSEDED. */
  onScan(ev: ScanEvent): VerificationResult[] {
    const out: VerificationResult[] = [];
    if (this.scan) {
      out.push(this.finalize('SUPERSEDED', null, 0, ev.ts));
    }
    this.scan = {
      scanId: ev.scanId,
      sku: ev.sku,
      expectedSide: ev.expectedSide,
      deadline: ev.ts + this.opts.scanTimeoutMs,
      candidate: null,
      candidateCount: 0,
    };
    this.tracks.clear();
    this.lastNoZoneTs = ev.ts;
    return out;
  }

  /** Per-frame hand observations (also serves as the clock tick). */
  onFrame(frame: HandFrame): VerificationResult[] {
    const out: VerificationResult[] = [];
    const ts = frame.ts;

    if (frame.hands.length > 0) this.lastHandsPresentTs = ts;
    if (frame.hands.some((h) => h.inBehindBand)) this.lastHandInBandTs = ts;
    if (frame.behindMotion) this.behindMotionSeenTs = ts;

    if (!this.scan) {
      this.updateTracks(frame, null);
      return out;
    }

    // 1. zone-exit candidates (dwell + retraction)
    this.updateTracks(frame, (side, dwell) => {
      if (dwell >= this.opts.dwellMs && this.opts.enabledSides.includes(side)) {
        this.scan!.candidate = { side, ts };
        this.scan!.candidateCount++;
      }
    });

    // 1b. a hand resting in a zone for a long time counts even without
    // retraction, so a worker leaning on the correct bin is not timed out
    for (const track of this.tracks.values()) {
      if (
        track.zone &&
        ts - track.enteredAt >= this.opts.dwellMs * 4 &&
        this.opts.enabledSides.includes(track.zone) &&
        this.scan.candidate?.side !== track.zone
      ) {
        this.scan.candidate = { side: track.zone, ts };
        this.scan.candidateCount++;
      }
    }

    // 2. BEHIND heuristic: hands gone after touching the edge band with motion
    if (
      this.opts.enabledSides.includes('BEHIND') &&
      frame.hands.length === 0 &&
      ts - this.lastHandsPresentTs >= this.opts.behindAbsenceMs &&
      this.lastHandInBandTs >= this.lastHandsPresentTs - 500 &&
      this.behindMotionSeenTs >= this.lastHandsPresentTs - 500 &&
      this.scan.candidate?.side !== 'BEHIND'
    ) {
      this.scan.candidate = { side: 'BEHIND', ts };
      this.scan.candidateCount++;
    }

    // 3. finalization
    const anyHandInZone = frame.hands.some((h) => h.zone !== null || h.inBehindBand);
    if (!anyHandInZone) {
      if (this.lastNoZoneTs === 0) this.lastNoZoneTs = ts;
    } else {
      this.lastNoZoneTs = 0;
    }

    const cand = this.scan.candidate;
    if (cand) {
      if (cand.side === this.scan.expectedSide) {
        // Correct side — no reason to wait for corrections.
        out.push(this.finalize('MATCH', cand.side, this.confidence(), ts));
        return out;
      }
      // Wrong side: give the worker a grace window to correct themselves.
      const settled =
        cand.side === 'BEHIND' ||
        (this.lastNoZoneTs !== 0 && ts - Math.max(this.lastNoZoneTs, cand.ts) >= this.opts.retractGraceMs);
      if (settled && ts - cand.ts >= this.opts.retractGraceMs) {
        out.push(this.finalize('MISMATCH', cand.side, this.confidence(), ts));
        return out;
      }
    }

    // 4. timeout
    if (ts >= this.scan.deadline) {
      const side = cand?.side ?? null;
      out.push(this.finalize(side ? 'MISMATCH' : 'TIMEOUT', side, side ? this.confidence() : 0, ts));
    }
    return out;
  }

  private confidence(): number {
    if (!this.scan) return 0;
    // Single clean candidate → high confidence; ambiguity lowers it.
    return this.scan.candidateCount <= 1 ? 0.9 : 0.6;
  }

  private updateTracks(
    frame: HandFrame,
    onZoneExit: ((side: 'LEFT' | 'RIGHT', dwellMs: number) => void) | null,
  ) {
    const seen = new Set<string>();
    for (const hand of frame.hands) {
      seen.add(hand.id);
      const track = this.tracks.get(hand.id);
      if (!track) {
        this.tracks.set(hand.id, { zone: hand.zone, enteredAt: frame.ts });
        continue;
      }
      if (track.zone !== hand.zone) {
        if (track.zone && onZoneExit) onZoneExit(track.zone, frame.ts - track.enteredAt);
        track.zone = hand.zone;
        track.enteredAt = frame.ts;
      }
    }
    // A hand that vanished while inside a zone also counts as leaving it.
    for (const [id, track] of this.tracks) {
      if (!seen.has(id)) {
        if (track.zone && onZoneExit) onZoneExit(track.zone, frame.ts - track.enteredAt);
        this.tracks.delete(id);
      }
    }
  }

  private finalize(
    outcome: Outcome,
    detectedSide: Side | null,
    confidence: number,
    ts: number,
  ): VerificationResult {
    const scan = this.scan!;
    this.scan = null;
    this.tracks.clear();
    return {
      type: 'VERIFICATION_RESULT',
      scanId: scan.scanId,
      sku: scan.sku,
      outcome,
      expectedSide: scan.expectedSide,
      detectedSide,
      confidence,
      ts,
    };
  }
}
