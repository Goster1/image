import { describe, expect, it } from 'vitest';
import {
  centroid,
  defaultRectPolygon,
  edgeBandToRect,
  pointInEdgeBand,
  pointInPolygon,
} from './geometry';

describe('pointInPolygon', () => {
  const rect = defaultRectPolygon(0.1, 0.2, 0.3, 0.4);

  it('detects points inside', () => {
    expect(pointInPolygon({ x: 0.2, y: 0.4 }, rect)).toBe(true);
  });

  it('detects points outside', () => {
    expect(pointInPolygon({ x: 0.5, y: 0.4 }, rect)).toBe(false);
    expect(pointInPolygon({ x: 0.2, y: 0.7 }, rect)).toBe(false);
  });

  it('works with a triangle', () => {
    const tri = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 0.5, y: 1 },
    ];
    expect(pointInPolygon({ x: 0.5, y: 0.5 }, tri)).toBe(true);
    expect(pointInPolygon({ x: 0.05, y: 0.9 }, tri)).toBe(false);
  });
});

describe('pointInEdgeBand', () => {
  it('bottom band', () => {
    const band = { edge: 'bottom' as const, thickness: 0.2 };
    expect(pointInEdgeBand({ x: 0.5, y: 0.9 }, band)).toBe(true);
    expect(pointInEdgeBand({ x: 0.5, y: 0.5 }, band)).toBe(false);
  });

  it('left band', () => {
    const band = { edge: 'left' as const, thickness: 0.15 };
    expect(pointInEdgeBand({ x: 0.1, y: 0.5 }, band)).toBe(true);
    expect(pointInEdgeBand({ x: 0.3, y: 0.5 }, band)).toBe(false);
  });
});

describe('edgeBandToRect', () => {
  it('right band covers the right strip', () => {
    expect(edgeBandToRect({ edge: 'right', thickness: 0.25 })).toEqual({
      x: 0.75,
      y: 0,
      w: 0.25,
      h: 1,
    });
  });
});

describe('centroid', () => {
  it('averages points', () => {
    expect(
      centroid([
        { x: 0, y: 0 },
        { x: 1, y: 1 },
      ]),
    ).toEqual({ x: 0.5, y: 0.5 });
  });
});
