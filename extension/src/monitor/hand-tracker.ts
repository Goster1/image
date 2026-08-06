/**
 * MediaPipe HandLandmarker wrapper. Wasm + model are bundled inside the
 * extension (MV3 forbids remote code); paths resolve via chrome.runtime.getURL.
 */
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import type { Point } from '../shared/geometry';
import { centroid } from '../shared/geometry';

export interface HandObservation {
  ts: number;
  hands: {
    /** 'Left' | 'Right' handedness label, suffixed with index for stability. */
    id: string;
    /** Palm centroid in normalized image coords. */
    palm: Point;
    /** All 21 landmarks (normalized) for the debug overlay. */
    landmarks: Point[];
  }[];
}

// Palm = wrist + base of each finger; stabler than fingertips while gripping.
const PALM_LANDMARKS = [0, 5, 9, 13, 17];

export class HandTracker {
  private landmarker: HandLandmarker | null = null;

  async init(): Promise<void> {
    const fileset = await FilesetResolver.forVisionTasks(chrome.runtime.getURL('wasm'));
    this.landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: chrome.runtime.getURL('models/hand_landmarker.task'),
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
      // A hand gripping a product is partially occluded — keep thresholds loose.
      minHandDetectionConfidence: 0.3,
      minHandPresenceConfidence: 0.3,
      minTrackingConfidence: 0.3,
    });
  }

  get ready(): boolean {
    return this.landmarker !== null;
  }

  detect(video: HTMLVideoElement, ts: number): HandObservation {
    if (!this.landmarker) return { ts, hands: [] };
    const result = this.landmarker.detectForVideo(video, ts);
    const hands: HandObservation['hands'] = [];
    for (let i = 0; i < result.landmarks.length; i++) {
      const lm = result.landmarks[i].map((p) => ({ x: p.x, y: p.y }));
      const label = result.handedness[i]?.[0]?.categoryName ?? `hand${i}`;
      hands.push({
        id: `${label}-${i}`,
        palm: centroid(PALM_LANDMARKS.map((j) => lm[j])),
        landmarks: lm,
      });
    }
    return { ts, hands };
  }

  close(): void {
    this.landmarker?.close();
    this.landmarker = null;
  }
}
