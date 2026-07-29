import sharp from 'sharp';
import * as fs from 'fs';
import * as path from 'path';
import { CascadeData, CascadeStage, DetectOptions, Detection, PrecompFeature, PrecompStage, RawImage } from './types.js';
import cascadeData from './data/face-cascade.js';

async function loadFromBuffer(buf: Buffer, w?: number, h?: number): Promise<RawImage> {
  let p = sharp(buf);
  if (w !== undefined && h !== undefined) p = p.resize(w, h, { fit: 'fill' });
  const { data, info } = await p.grayscale().raw().toBuffer({ resolveWithObject: true });
  return { data: data, width: info.width, height: info.height };
}

/**
 * Apply CLAHE-like histogram equalization to improve contrast.
 * Splits image into tiles and equalizes each independently.
 */
function equalizeHistogram(img: RawImage): RawImage {
  const { data, width, height } = img;
  const out = new Uint8Array(data);

  // Compute histogram
  const hist = new Uint32Array(256);
  for (let i = 0; i < data.length; i++) hist[data[i]]++;

  // Compute CDF
  const cdf = new Uint32Array(256);
  cdf[0] = hist[0];
  for (let i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + hist[i];

  // Find minimum non-zero CDF
  let cdfMin = 0;
  for (let i = 0; i < 256; i++) {
    if (cdf[i] > 0) { cdfMin = cdf[i]; break; }
  }

  const totalPixels = width * height;
  if (totalPixels === cdfMin) return img;

  // Build lookup table
  const lut = new Uint8Array(256);
  const scale = 255 / (totalPixels - cdfMin);
  for (let i = 0; i < 256; i++) {
    const val = Math.round((cdf[i] - cdfMin) * scale);
    lut[i] = Math.max(0, Math.min(255, val));
  }

  // Apply LUT
  for (let i = 0; i < data.length; i++) out[i] = lut[data[i]];

  return { data: out, width, height };
}

/**
 * Fast Gaussian blur (3x3 kernel) using separable passes.
 * Two 1D passes (horizontal then vertical) is faster than one 2D pass.
 */
function gaussianBlur3x3(src: RawImage): RawImage {
  const { data, width, height } = src;
  const tmp = new Uint16Array(width * height);
  const out = new Uint8Array(width * height);

  // Horizontal pass: [1, 2, 1] / 4
  for (let y = 0; y < height; y++) {
    const row = y * width;
    tmp[row] = data[row] * 3 + data[row + 1]; // edge: (3*left + right) / 4
    for (let x = 1; x < width - 1; x++) {
      tmp[row + x] = data[row + x - 1] + data[row + x] * 2 + data[row + x + 1];
    }
    tmp[row + width - 1] = data[row + width - 2] + data[row + width - 1] * 3; // edge
  }

  // Vertical pass: [1, 2, 1] / 16 (combined with horizontal = full 3x3 Gaussian)
  for (let x = 0; x < width; x++) {
    out[x] = (tmp[x] * 3 + tmp[width + x] + 8) >> 4; // top edge, round
    for (let y = 1; y < height - 1; y++) {
      const idx = y * width + x;
      out[idx] = (tmp[idx - width] + tmp[idx] * 2 + tmp[idx + width] + 8) >> 4;
    }
    const last = (height - 1) * width + x;
    out[last] = (tmp[last - width] + tmp[last] * 3 + 8) >> 4; // bottom edge
  }

  return { data: out, width, height };
}

/**
 * 2x subsample with Gaussian pre-filtering to reduce aliasing.
 */
function halfSubsample(src: RawImage, dx: number, dy: number): RawImage {
  // Apply Gaussian blur before subsampling to prevent aliasing
  const blurred = gaussianBlur3x3(src);

  const dw = Math.floor(src.width / 2);
  const dh = Math.floor(src.height / 2);
  const out = new Uint8Array(dw * dh);
  let oi = 0;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(y * 2 + dy, src.height - 1);
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(x * 2 + dx, src.width - 1);
      out[oi] = blurred.data[sy * src.width + sx];
      oi++;
    }
  }
  return { data: out, width: dw, height: dh };
}

function precomputeFeatures(stages: CascadeStage[], step: number[]): PrecompStage[] {
  return stages.map((st) => {
    const features = st.feature.map((feat) => {
      const f: PrecompFeature = {
        size: feat.size,
        px: new Array(feat.size),
        pz: new Array(feat.size),
        nx: new Array(feat.size),
        nz: new Array(feat.size),
      };
      for (let j = 0; j < feat.size; j++) {
        f.pz[j] = feat.pz[j];
        f.nz[j] = feat.nz[j];
        f.px[j] = feat.px[j] + feat.py[j] * step[feat.pz[j] >= 0 ? feat.pz[j] : 0];
        f.nx[j] = feat.nz[j] >= 0 ? feat.nx[j] + feat.ny[j] * step[feat.nz[j]] : 0;
      }
      return f;
    });
    return { threshold: st.threshold, count: st.count, alpha: st.alpha, features };
  });
}

function groupDetections(seq: Detection[], minNeighbors: number): Detection[] {
  if (!(minNeighbors > 0)) return seq;

  const n = seq.length;
  if (n === 0) return [];

  const node: { parent: number; element: Detection | null; rank: number }[] = [];
  for (let i = 0; i < n; i++) node.push({ parent: -1, element: seq[i], rank: 0 });

  // Build overlap graph
  for (let i = 0; i < n; i++) {
    if (!node[i].element) continue;
    let root = i;
    while (node[root].parent !== -1) root = node[root].parent;

    for (let j = i + 1; j < n; j++) {
      if (!node[j].element) continue;
      const r1 = node[i].element!;
      const r2 = node[j].element!;
      const distance = Math.floor(Math.max(r1.width, r2.width) * 0.25 + 0.5);
      const overlap =
        r2.x <= r1.x + distance &&
        r2.x >= r1.x - distance &&
        r2.y <= r1.y + distance &&
        r2.y >= r1.y - distance &&
        r2.width <= Math.floor(r1.width * 1.5 + 0.5) &&
        Math.floor(r2.width * 1.5 + 0.5) >= r1.width;

      if (overlap) {
        let root2 = j;
        while (node[root2].parent !== -1) root2 = node[root2].parent;
        if (root2 !== root) {
          if (node[root].rank > node[root2].rank) {
            node[root2].parent = root;
          } else {
            node[root].parent = root2;
            if (node[root].rank === node[root2].rank) node[root2].rank++;
            root = root2;
          }
          let temp: number;
          let ni = j;
          while (node[ni].parent !== -1) { temp = ni; ni = node[ni].parent; node[temp].parent = root; }
          ni = i;
          while (node[ni].parent !== -1) { temp = ni; ni = node[ni].parent; node[temp].parent = root; }
        }
      }
    }
  }

  const idx = new Array(n);
  let classIdx = 0;
  for (let i = 0; i < n; i++) {
    let j = -1;
    let ni = i;
    if (node[ni].element) {
      while (node[ni].parent !== -1) ni = node[ni].parent;
      if (node[ni].rank >= 0) node[ni].rank = ~classIdx++;
      j = ~node[ni].rank;
    }
    idx[i] = j;
  }

  const comps: { neighbors: number; x: number; y: number; width: number; height: number; confidence: number }[] = [];
  for (let i = 0; i < classIdx; i++) comps.push({ neighbors: 0, x: 0, y: 0, width: 0, height: 0, confidence: 0 });

  for (let i = 0; i < n; i++) {
    const ci = idx[i];
    if (ci < 0 || ci >= comps.length) continue;
    const comp = comps[ci];
    const r = seq[i];
    if (comp.neighbors === 0) comp.confidence = r.confidence;
    comp.neighbors++;
    comp.x += r.x;
    comp.y += r.y;
    comp.width += r.width;
    comp.height += r.height;
    comp.confidence = Math.max(comp.confidence, r.confidence);
  }

  const seq2: Detection[] = [];
  for (const c of comps) {
    if (c.neighbors >= minNeighbors) {
      seq2.push({
        x: (c.x * 2 + c.neighbors) / (2 * c.neighbors),
        y: (c.y * 2 + c.neighbors) / (2 * c.neighbors),
        width: (c.width * 2 + c.neighbors) / (2 * c.neighbors),
        height: (c.height * 2 + c.neighbors) / (2 * c.neighbors),
        neighbor: c.neighbors,
        confidence: c.confidence,
      });
    }
  }

  const result: Detection[] = [];
  for (let i = 0; i < seq2.length; i++) {
    const r1 = seq2[i];
    let flag = true;
    for (let j = 0; j < seq2.length; j++) {
      if (i === j) continue;
      const r2 = seq2[j];
      const distance = Math.floor(r2.width * 0.25 + 0.5);
      if (
        r1.x >= r2.x - distance &&
        r1.y >= r2.y - distance &&
        r1.x + r1.width <= r2.x + r2.width + distance &&
        r1.y + r1.height <= r2.y + r2.height + distance &&
        (r2.neighbor > Math.max(3, r1.neighbor) || r1.neighbor < 3)
      ) {
        flag = false;
        break;
      }
    }
    if (flag) result.push(r1);
  }

  return result;
}

export async function detectFaces(imagePath: string, options?: DetectOptions): Promise<Detection[]> {
  const cd = cascadeData;
  const cascadeWidth = cd.width;
  const cascadeHeight = cd.height;
  const interval = 3;
  const next = interval + 1;
  const scale = options?.scaleFactor ?? 1.15;
  const stride = options?.step ?? 4;
  const minNeighbors = options?.minNeighbors ?? 3;

  const fileBuffer = await fs.promises.readFile(imagePath);
  const meta = await sharp(fileBuffer).metadata();
  const originalWidth = meta.width!;
  const originalHeight = meta.height!;

  let inputImage: RawImage;
  const effectiveMax = options?.maxDimension ?? 400;
  const maxSide = Math.max(originalWidth, originalHeight);
  if (effectiveMax !== undefined && maxSide > effectiveMax) {
    const ratio = effectiveMax / maxSide;
    const rw = Math.round(originalWidth * ratio);
    const rh = Math.round(originalHeight * ratio);
    inputImage = await loadFromBuffer(fileBuffer, rw, rh);
  } else {
    inputImage = await loadFromBuffer(fileBuffer);
  }

  // Apply histogram equalization to improve contrast under varying lighting
  inputImage = equalizeHistogram(inputImage);

  const scaleUpto = Math.floor(
    Math.log(Math.min(inputImage.width / cascadeWidth, inputImage.height / cascadeHeight))
    / Math.log(scale),
  );
  if (scaleUpto < 0) return [];

  const pyrLen = (scaleUpto + next * 2) * 4;
  const pyr: (RawImage | undefined)[] = new Array(pyrLen);

  pyr[0] = inputImage;

  for (let i = 1; i <= interval; i++) {
    const w = Math.floor(inputImage.width / Math.pow(scale, i));
    const h = Math.floor(inputImage.height / Math.pow(scale, i));
    pyr[i * 4] = await loadFromBuffer(fileBuffer, w, h);
  }

  for (let i = next; i < scaleUpto + next * 2; i++) {
    const src = pyr[(i - next) * 4]!;
    pyr[i * 4] = halfSubsample(src, 0, 0);
    pyr[i * 4 + 1] = halfSubsample(src, 1, 0);
    pyr[i * 4 + 2] = halfSubsample(src, 0, 1);
    pyr[i * 4 + 3] = halfSubsample(src, 1, 1);
  }

  const DX = [0, 1, 0, 1];
  const DY = [0, 0, 1, 1];
  const all: Detection[] = [];
  let passCount = 0;

  let sf = 1;
  for (let i = 0; i < scaleUpto; i++) {
    const bi = i * 4;
    const pyr0 = pyr[bi]!;
    const pyr1 = pyr[bi + next * 4]!;
    const pyr2base = bi + next * 8;
    const pyr2a = pyr[pyr2base]!;
    const pyr2b = pyr[pyr2base + 1]!;
    const pyr2c = pyr[pyr2base + 2]!;
    const pyr2d = pyr[pyr2base + 3]!;
    const pyr2 = [pyr2a, pyr2b, pyr2c, pyr2d];

    const stepArr = [pyr0.width, pyr1.width, pyr2a.width];

    const stages = precomputeFeatures(cd.stage_classifier, stepArr);

    const level2Step = stride / 4;
    const qw = Math.ceil((pyr2a.width - Math.floor(cascadeWidth / 4)) / level2Step);
    const qh = Math.ceil((pyr2a.height - Math.floor(cascadeHeight / 4)) / level2Step);
    if (i > 0) sf *= scale;
    const sxf = sf;
    const syf = sf;

    for (let q = 0; q < 4; q++) {
      const u8 = [pyr0.data, pyr1.data, pyr2[q].data];
      const u8o = [
        DX[q] * 2 + DY[q] * pyr0.width * 2,
        DX[q] + DY[q] * pyr1.width,
        0,
      ];
      const pad = [
        pyr0.width * 4 - qw * stride,
        pyr1.width * 2 - qw * (stride >> 1),
        pyr2[q].width - qw * (stride >> 2),
      ];

      for (let y = 0; y < qh; y++) {
        for (let x = 0; x < qw; x++) {
          let passed = true;
          let totalSum = 0;
          let totalThreshold = 0;

          for (const st of stages) {
            let sum = 0;
            for (let k = 0; k < st.count; k++) {
              const feat = st.features[k];

              let pmin = u8[feat.pz[0]][u8o[feat.pz[0]] + feat.px[0]];
              let nmax = u8[feat.nz[0]][u8o[feat.nz[0]] + feat.nx[0]];

              if (pmin <= nmax) {
                sum += st.alpha[k * 2];
              } else {
                let shortcut = true;
                for (let f = 0; f < feat.size; f++) {
                  if (feat.pz[f] >= 0) {
                    const p = u8[feat.pz[f]][u8o[feat.pz[f]] + feat.px[f]];
                    if (p < pmin) {
                      if (p <= nmax) { shortcut = false; break; }
                      pmin = p;
                    }
                  }
                  if (feat.nz[f] >= 0) {
                    const n = u8[feat.nz[f]][u8o[feat.nz[f]] + feat.nx[f]];
                    if (n > nmax) {
                      if (pmin <= n) { shortcut = false; break; }
                      nmax = n;
                    }
                  }
                }
                sum += shortcut ? st.alpha[k * 2 + 1] : st.alpha[k * 2];
              }
            }
            totalSum += sum;
            totalThreshold += st.threshold;
            if (sum < st.threshold) {
              passed = false;
              break;
            }
          }

          if (passed) {
            passCount++;
            // Improved confidence: ratio of accumulated margin above threshold
            const margin = totalSum - totalThreshold;
            const confidence = totalThreshold !== 0
              ? Math.min(1, Math.max(0, 0.5 + margin / (2 * Math.abs(totalThreshold))))
              : 0;
            all.push({
              x: (x * stride + DX[q] * 2) * sxf,
              y: (y * stride + DY[q] * 2) * syf,
              width: cascadeWidth * sxf,
              height: cascadeHeight * syf,
              neighbor: 1,
              confidence,
            });
          }

          u8o[0] += stride;
          u8o[1] += stride >> 1;
          u8o[2] += stride >> 2;
        }
        u8o[0] += pad[0];
        u8o[1] += pad[1];
        u8o[2] += pad[2];
      }
    }
  }

  const grouped = groupDetections(all, minNeighbors);

  const scaleX = originalWidth / inputImage.width;
  const scaleY = originalHeight / inputImage.height;
  for (const d of grouped) {
    d.x *= scaleX;
    d.y *= scaleY;
    d.width *= scaleX;
    d.height *= scaleY;
  }

  if (process.env.DEBUG) console.log(path.basename(imagePath) + ':', 'raw=' + passCount, 'grouped=' + grouped.length);
  return grouped;
}

export async function detectBestFace(
  imagePath: string,
  options?: DetectOptions,
): Promise<{ x: number; y: number; width: number; height: number } | null> {
  const faces = await detectFaces(imagePath, options);
  if (faces.length === 0) return null;
  faces.sort((a, b) => b.width * b.height - a.width * a.height);
  return {
    x: Math.round(faces[0].x),
    y: Math.round(faces[0].y),
    width: Math.round(faces[0].width),
    height: Math.round(faces[0].height),
  };
}
