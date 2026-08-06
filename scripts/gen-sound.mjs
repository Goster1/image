// Generates the mismatch alert sound (double beep) as a small WAV file,
// avoiding any binary asset dependencies in the repo.
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outFile = resolve(root, 'extension/public/sounds/alert.wav');
mkdirSync(dirname(outFile), { recursive: true });

const sampleRate = 22050;
const durationS = 0.7;
const n = Math.floor(sampleRate * durationS);
const data = Buffer.alloc(n * 2);

for (let i = 0; i < n; i++) {
  const t = i / sampleRate;
  // two 0.25 s beeps (880 Hz then 660 Hz) with a short gap and fade-out
  let amp = 0;
  if (t < 0.25) amp = Math.sin(2 * Math.PI * 880 * t) * (1 - t / 0.3);
  else if (t >= 0.35 && t < 0.6)
    amp = Math.sin(2 * Math.PI * 660 * t) * (1 - (t - 0.35) / 0.3);
  data.writeInt16LE(Math.round(amp * 0.6 * 32767), i * 2);
}

const header = Buffer.alloc(44);
header.write('RIFF', 0);
header.writeUInt32LE(36 + data.length, 4);
header.write('WAVE', 8);
header.write('fmt ', 12);
header.writeUInt32LE(16, 16);
header.writeUInt16LE(1, 20); // PCM
header.writeUInt16LE(1, 22); // mono
header.writeUInt32LE(sampleRate, 24);
header.writeUInt32LE(sampleRate * 2, 28);
header.writeUInt16LE(2, 32);
header.writeUInt16LE(16, 34);
header.write('data', 36);
header.writeUInt32LE(data.length, 40);

writeFileSync(outFile, Buffer.concat([header, data]));
console.log(`wrote ${outFile}`);
