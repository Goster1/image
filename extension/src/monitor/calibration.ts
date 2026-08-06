/**
 * Zone calibration editor: draws LEFT/RIGHT polygons and the BEHIND edge
 * band on a canvas over the live video, and lets the operator drag vertices
 * or whole polygons. All coordinates are normalized (0..1).
 */
import type { EdgeBand, Point, Polygon } from '../shared/geometry';
import { edgeBandToRect, pointInPolygon } from '../shared/geometry';
import type { StationConfig } from '../shared/config';
import type { Side } from '../shared/messages';
import { SIDE_LABELS_SK } from '../shared/messages';

export const ZONE_COLORS: Record<Side, string> = {
  LEFT: '#dc2626',
  RIGHT: '#2563eb',
  BEHIND: '#ea580c',
};

const VERTEX_HIT_PX = 12;

type DragState =
  | { kind: 'vertex'; side: 'LEFT' | 'RIGHT'; index: number }
  | { kind: 'polygon'; side: 'LEFT' | 'RIGHT'; start: Point; original: Polygon }
  | { kind: 'band' }
  | null;

export class CalibrationEditor {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private zones: StationConfig['zones'];
  private drag: DragState = null;
  private editable = false;
  private mirrored = false;
  /** Latest hand palm positions for live feedback. */
  private handDots: Point[] = [];
  private onChange: () => void;

  constructor(canvas: HTMLCanvasElement, zones: StationConfig['zones'], onChange: () => void) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.zones = zones;
    this.onChange = onChange;
    canvas.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    canvas.addEventListener('pointermove', (e) => this.onPointerMove(e));
    canvas.addEventListener('pointerup', () => (this.drag = null));
    canvas.addEventListener('pointercancel', () => (this.drag = null));
  }

  setZones(zones: StationConfig['zones']): void {
    this.zones = zones;
  }

  /** The preview may be CSS-flipped; pointer X must be un-flipped to match. */
  setMirrored(mirrored: boolean): void {
    this.mirrored = mirrored;
  }

  setEditable(editable: boolean): void {
    this.editable = editable;
    this.canvas.style.cursor = editable ? 'crosshair' : 'default';
  }

  setHandDots(dots: Point[]): void {
    this.handDots = dots;
  }

  private toNorm(e: PointerEvent): Point {
    const rect = this.canvas.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
    return {
      x: this.mirrored ? 1 - x : x,
      y: Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
    };
  }

  private onPointerDown(e: PointerEvent): void {
    if (!this.editable) return;
    this.canvas.setPointerCapture(e.pointerId);
    const p = this.toNorm(e);
    const rect = this.canvas.getBoundingClientRect();

    // vertex hit first (takes precedence over polygon body)
    for (const side of ['LEFT', 'RIGHT'] as const) {
      const poly = this.zones[side];
      if (!poly) continue;
      for (let i = 0; i < poly.length; i++) {
        const dx = (poly[i].x - p.x) * rect.width;
        const dy = (poly[i].y - p.y) * rect.height;
        if (Math.hypot(dx, dy) <= VERTEX_HIT_PX) {
          this.drag = { kind: 'vertex', side, index: i };
          return;
        }
      }
    }
    for (const side of ['LEFT', 'RIGHT'] as const) {
      const poly = this.zones[side];
      if (poly && pointInPolygon(p, poly)) {
        this.drag = { kind: 'polygon', side, start: p, original: poly.map((v) => ({ ...v })) };
        return;
      }
    }
    if (this.zones.BEHIND) {
      const r = edgeBandToRect(this.zones.BEHIND);
      if (p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h) {
        this.drag = { kind: 'band' };
      }
    }
  }

  private onPointerMove(e: PointerEvent): void {
    if (!this.drag) return;
    const p = this.toNorm(e);
    if (this.drag.kind === 'vertex') {
      const poly = this.zones[this.drag.side];
      if (poly) poly[this.drag.index] = p;
    } else if (this.drag.kind === 'polygon') {
      const dx = p.x - this.drag.start.x;
      const dy = p.y - this.drag.start.y;
      const poly = this.zones[this.drag.side];
      if (poly) {
        for (let i = 0; i < poly.length; i++) {
          poly[i] = {
            x: Math.min(1, Math.max(0, this.drag.original[i].x + dx)),
            y: Math.min(1, Math.max(0, this.drag.original[i].y + dy)),
          };
        }
      }
    } else if (this.drag.kind === 'band' && this.zones.BEHIND) {
      const band = this.zones.BEHIND;
      // Dragging adjusts thickness from the band's edge.
      switch (band.edge) {
        case 'left':
          band.thickness = p.x;
          break;
        case 'right':
          band.thickness = 1 - p.x;
          break;
        case 'top':
          band.thickness = p.y;
          break;
        case 'bottom':
          band.thickness = 1 - p.y;
          break;
      }
      band.thickness = Math.min(0.5, Math.max(0.05, band.thickness));
    }
    this.onChange();
  }

  /** Draw zones + live hand dots. Call once per rendered frame. */
  render(activeZones: Set<Side>): void {
    const { width: w, height: h } = this.canvas;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);

    for (const side of ['LEFT', 'RIGHT'] as const) {
      const poly = this.zones[side];
      if (!poly) continue;
      const color = ZONE_COLORS[side];
      ctx.beginPath();
      ctx.moveTo(poly[0].x * w, poly[0].y * h);
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i].x * w, poly[i].y * h);
      ctx.closePath();
      ctx.fillStyle = color + (activeZones.has(side) ? '66' : '2e');
      ctx.fill();
      ctx.strokeStyle = color;
      ctx.lineWidth = activeZones.has(side) ? 4 : 2;
      ctx.stroke();
      this.drawLabel(SIDE_LABELS_SK[side], poly[0].x * w + 6, poly[0].y * h + 22, color);
      if (this.editable) {
        for (const v of poly) {
          ctx.beginPath();
          ctx.arc(v.x * w, v.y * h, 6, 0, Math.PI * 2);
          ctx.fillStyle = '#fff';
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    }

    if (this.zones.BEHIND) {
      const r = edgeBandToRect(this.zones.BEHIND);
      const color = ZONE_COLORS.BEHIND;
      ctx.fillStyle = color + (activeZones.has('BEHIND') ? '66' : '2e');
      ctx.fillRect(r.x * w, r.y * h, r.w * w, r.h * h);
      ctx.strokeStyle = color;
      ctx.lineWidth = activeZones.has('BEHIND') ? 4 : 2;
      ctx.strokeRect(r.x * w, r.y * h, r.w * w, r.h * h);
      this.drawLabel(SIDE_LABELS_SK.BEHIND, r.x * w + 6, r.y * h + 22, color);
    }

    for (const dot of this.handDots) {
      ctx.beginPath();
      ctx.arc(dot.x * w, dot.y * h, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#22c55e';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  private drawLabel(text: string, x: number, y: number, color: string): void {
    const ctx = this.ctx;
    ctx.font = 'bold 16px system-ui';
    ctx.fillStyle = color;
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 4;
    ctx.strokeText(text, x, y);
    ctx.fillText(text, x, y);
  }
}

export function bandRect(band: EdgeBand): { x: number; y: number; w: number; h: number } {
  return edgeBandToRect(band);
}
