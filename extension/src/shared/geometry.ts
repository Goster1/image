/**
 * Zone geometry in normalized image coordinates (0..1 for both axes),
 * so calibration is independent of the camera resolution.
 */

export interface Point {
  x: number;
  y: number;
}

/** Closed polygon, at least 3 vertices, normalized coords. */
export type Polygon = Point[];

/** Band along one frame edge (used for the BEHIND heuristic). */
export interface EdgeBand {
  edge: 'left' | 'right' | 'bottom' | 'top';
  /** Band thickness as a fraction of the frame (0..0.5). */
  thickness: number;
}

export function pointInPolygon(p: Point, poly: Polygon): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i];
    const b = poly[j];
    if (
      a.y > p.y !== b.y > p.y &&
      p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x
    ) {
      inside = !inside;
    }
  }
  return inside;
}

export function pointInEdgeBand(p: Point, band: EdgeBand): boolean {
  switch (band.edge) {
    case 'left':
      return p.x <= band.thickness;
    case 'right':
      return p.x >= 1 - band.thickness;
    case 'top':
      return p.y <= band.thickness;
    case 'bottom':
      return p.y >= 1 - band.thickness;
  }
}

export function centroid(points: Point[]): Point {
  let x = 0;
  let y = 0;
  for (const p of points) {
    x += p.x;
    y += p.y;
  }
  return { x: x / points.length, y: y / points.length };
}

export function edgeBandToRect(band: EdgeBand): { x: number; y: number; w: number; h: number } {
  switch (band.edge) {
    case 'left':
      return { x: 0, y: 0, w: band.thickness, h: 1 };
    case 'right':
      return { x: 1 - band.thickness, y: 0, w: band.thickness, h: 1 };
    case 'top':
      return { x: 0, y: 0, w: 1, h: band.thickness };
    case 'bottom':
      return { x: 0, y: 1 - band.thickness, w: 1, h: band.thickness };
  }
}

export function defaultRectPolygon(x: number, y: number, w: number, h: number): Polygon {
  return [
    { x, y },
    { x: x + w, y },
    { x: x + w, y: y + h },
    { x, y: y + h },
  ];
}
