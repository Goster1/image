// Generates the extension toolbar icons (simple camera glyph on indigo)
// without any image dependencies — raw PNG encoding via zlib.
import { deflateSync } from 'node:zlib';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = resolve(root, 'extension/public/icons');
mkdirSync(outDir, { recursive: true });

function crc32(buf) {
  let c,
    table = [];
  for (let n = 0; n < 256; n++) {
    c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  let crc = 0xffffffff;
  for (const b of buf) crc = table[(crc ^ b) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
}

function png(size, pixelFn) {
  const raw = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) {
    const row = y * (size * 4 + 1);
    raw[row] = 0; // filter: none
    for (let x = 0; x < size; x++) {
      const [r, g, b, a] = pixelFn(x, y, size);
      const o = row + 1 + x * 4;
      raw[o] = r;
      raw[o + 1] = g;
      raw[o + 2] = b;
      raw[o + 3] = a;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // RGBA
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

// Indigo rounded square with a white camera-lens circle and red REC dot.
function pixel(x, y, size) {
  const u = x / size,
    v = y / size;
  const corner = 0.12;
  const inCorner =
    (u < corner || u > 1 - corner) &&
    (v < corner || v > 1 - corner) &&
    Math.hypot(
      Math.min(u, 1 - u) - corner,
      Math.min(v, 1 - v) - corner,
    ) > corner;
  if (inCorner) return [0, 0, 0, 0];
  const dLens = Math.hypot(u - 0.5, v - 0.55);
  if (dLens < 0.2) return [255, 255, 255, 255];
  if (dLens < 0.27) return [199, 210, 254, 255];
  const dRec = Math.hypot(u - 0.78, v - 0.22);
  if (dRec < 0.09) return [239, 68, 68, 255];
  return [79, 70, 229, 255];
}

for (const size of [16, 48, 128]) {
  const file = resolve(outDir, `icon${size}.png`);
  writeFileSync(file, png(size, pixel));
  console.log(`wrote ${file}`);
}
