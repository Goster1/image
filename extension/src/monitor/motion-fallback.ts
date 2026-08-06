/**
 * Frame-difference motion detector on a downscaled canvas. Serves two roles:
 *  - corroborates the BEHIND heuristic (motion in the edge band),
 *  - full fallback detection mode when MediaPipe wasm fails to load.
 */
import type { EdgeBand, Polygon } from '../shared/geometry';
import { edgeBandToRect, pointInPolygon } from '../shared/geometry';

const W = 160;
const H = 120;

export interface MotionSample {
  ts: number;
  /** Mean absolute pixel diff (0..255) per zone key. */
  zoneEnergy: Record<string, number>;
}

export class MotionDetector {
  private canvas: OffscreenCanvas;
  private ctx: OffscreenCanvasRenderingContext2D;
  private prev: Uint8ClampedArray | null = null;
  /** Per-zone bitmask of which downscaled pixels belong to the zone. */
  private masks = new Map<string, Uint8Array>();

  constructor() {
    this.canvas = new OffscreenCanvas(W, H);
    this.ctx = this.canvas.getContext('2d', { willReadFrequently: true })!;
  }

  setZones(zones: Record<string, Polygon | EdgeBand>): void {
    this.masks.clear();
    for (const [key, zone] of Object.entries(zones)) {
      const mask = new Uint8Array(W * H);
      if (Array.isArray(zone)) {
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            if (pointInPolygon({ x: (x + 0.5) / W, y: (y + 0.5) / H }, zone)) {
              mask[y * W + x] = 1;
            }
          }
        }
      } else {
        const r = edgeBandToRect(zone);
        const x0 = Math.floor(r.x * W);
        const x1 = Math.ceil((r.x + r.w) * W);
        const y0 = Math.floor(r.y * H);
        const y1 = Math.ceil((r.y + r.h) * H);
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) mask[y * W + x] = 1;
        }
      }
      this.masks.set(key, mask);
    }
  }

  sample(video: HTMLVideoElement, ts: number): MotionSample {
    this.ctx.drawImage(video, 0, 0, W, H);
    const frame = this.ctx.getImageData(0, 0, W, H).data;
    const zoneEnergy: Record<string, number> = {};
    if (this.prev) {
      for (const [key, mask] of this.masks) {
        let sum = 0;
        let count = 0;
        for (let i = 0; i < W * H; i++) {
          if (!mask[i]) continue;
          const o = i * 4;
          // luma approximation is enough for motion energy
          const cur = (frame[o] + frame[o + 1] + frame[o + 2]) / 3;
          const was = (this.prev[o] + this.prev[o + 1] + this.prev[o + 2]) / 3;
          sum += Math.abs(cur - was);
          count++;
        }
        zoneEnergy[key] = count > 0 ? sum / count : 0;
      }
    }
    this.prev = new Uint8ClampedArray(frame);
    return { ts, zoneEnergy };
  }

  reset(): void {
    this.prev = null;
  }
}
