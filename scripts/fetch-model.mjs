// Downloads the MediaPipe hand landmarker model into extension/public/models/.
// Run once after `npm install` (also invoked automatically by `npm run build`
// only if the model is missing — see README).
import { mkdirSync, existsSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dest = resolve(root, 'extension/public/models/hand_landmarker.task');

if (existsSync(dest)) {
  console.log(`Model already present: ${dest}`);
  process.exit(0);
}

console.log(`Downloading ${MODEL_URL} ...`);
const res = await fetch(MODEL_URL);
if (!res.ok) {
  console.error(`Download failed: HTTP ${res.status}`);
  process.exit(1);
}
const buf = Buffer.from(await res.arrayBuffer());
mkdirSync(dirname(dest), { recursive: true });
writeFileSync(dest, buf);
console.log(`Saved ${buf.length} bytes to ${dest}`);
