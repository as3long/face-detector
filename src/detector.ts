import sharp from 'sharp';
import * as fs from 'fs';
import { CascadeData, CascadeStage, DetectOptions, Detection, PrecompFeature, PrecompStage, RawImage } from './types.js';
import cascadeData from './data/face-cascade.json';

function grayscaleInPlace(data: Uint8Array): void {
  for (let i = 0; i < data.length; i += 4) {
    const g = (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114) | 0;
    data[i] = g;
    data[i + 1] = g;
    data[i + 2] = g;
  }
}

async function loadFromBuffer(buf: Buffer, w?: number, h?: number): Promise<RawImage> {
  let p = sharp(buf);
  if (w !== undefined && h !== undefined) p = p.resize(w, h, { fit: 'fill' });
  const { data, info } = await p.raw().toBuffer({ resolveWithObject: true });
  const arr = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  grayscaleInPlace(arr);
  return { data: arr, width: info.width, height: info.height };
}

function halfSubsample(src: RawImage, dx: number, dy: number): RawImage {
  const dw = Math.floor(src.width / 2);
  const dh = Math.floor(src.height / 2);
  const out = new Uint8Array(dw * dh * 4);
  let oi = 0;
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(y * 2 + dy, src.height - 1);
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(x * 2 + dx, src.width - 1);
      const si = (sy * src.width + sx) * 4;
      out[oi] = src.data[si];
      out[oi + 1] = src.data[si + 1];
      out[oi + 2] = src.data[si + 2];
      out[oi + 3] = src.data[si + 3];
      oi += 4;
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
        f.px[j] = feat.px[j] * 4 + feat.py[j] * step[feat.pz[j] >= 0 ? feat.pz[j] : 0];
        f.nx[j] = feat.nz[j] >= 0 ? feat.nx[j] * 4 + feat.ny[j] * step[feat.nz[j]] : 0;
      }
      return f;
    });
    return { threshold: st.threshold, count: st.count, alpha: st.alpha, features };
  });
}

function groupDetections(seq: Detection[], minNeighbors: number): Detection[] {
  if (!(minNeighbors > 0)) return seq;

  const n = seq.length;
  const node: { parent: number; element: Detection | null; rank: number }[] = [];
  for (let i = 0; i < n; i++) node.push({ parent: -1, element: seq[i], rank: 0 });

  for (let i = 0; i < n; i++) {
    if (!node[i].element) continue;
    let root = i;
    while (node[root].parent !== -1) root = node[root].parent;

    for (let j = 0; j < n; j++) {
      if (i === j || !node[j].element) continue;
      const r1 = node[i].element!;
      const r2 = node[j].element!;
      const distance = Math.floor(r1.width * 0.25 + 0.5);
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
  const cd = cascadeData as unknown as CascadeData;
  const cascadeWidth = cd.width;
  const cascadeHeight = cd.height;
  const interval = 3;
  const next = interval + 1;
  const scale = options?.scaleFactor ?? Math.pow(2, 1 / (interval + 1));
  const stride = options?.step ?? 4;
  const minNeighbors = options?.minNeighbors ?? 3;

  const fileBuffer = await fs.promises.readFile(imagePath);
  const meta = await sharp(fileBuffer).metadata();
  const originalWidth = meta.width!;
  const originalHeight = meta.height!;

  const effectiveMax = Math.min(options?.maxDimension ?? 200, 500);
  const maxSide = Math.max(originalWidth, originalHeight);
  let inputImage: RawImage;
  if (maxSide > effectiveMax) {
    const ratio = effectiveMax / maxSide;
    const rw = Math.round(originalWidth * ratio);
    const rh = Math.round(originalHeight * ratio);
    inputImage = await loadFromBuffer(fileBuffer, rw, rh);
  } else {
    inputImage = await loadFromBuffer(fileBuffer);
  }

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

    const stepArr = [pyr0.width * 4, pyr1.width * 4, pyr2a.width * 4];

    const stages = precomputeFeatures(cd.stage_classifier, stepArr);

    const level2Step = stride / 4;
    const qw = Math.ceil((pyr2a.width - Math.floor(cascadeWidth / 4)) / level2Step);
    const qh = Math.ceil((pyr2a.height - Math.floor(cascadeHeight / 4)) / level2Step);
    const sxf = Math.pow(scale, i);
    const syf = Math.pow(scale, i);

    for (let q = 0; q < 4; q++) {
      const u8 = [pyr0.data, pyr1.data, pyr2[q].data];
      const u8o = [
        DX[q] * 8 + DY[q] * pyr0.width * 8,
        DX[q] * 4 + DY[q] * pyr1.width * 4,
        0,
      ];
      const pad = [
        pyr0.width * 4 - qw * stride * 4,
        pyr1.width * 4 - qw * stride * 2,
        pyr2[q].width * 4 - qw * stride,
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
            const confidence = totalThreshold > 0 ? Math.min(1, totalSum / totalThreshold) : 0;
            all.push({
              x: (x * stride + DX[q] * 2) * sxf,
              y: (y * stride + DY[q] * 2) * syf,
              width: cascadeWidth * sxf,
              height: cascadeHeight * syf,
              neighbor: 1,
              confidence,
            });
          }

          u8o[0] += stride * 4;
          u8o[1] += stride * 2;
          u8o[2] += stride;
        }
        u8o[0] += pad[0];
        u8o[1] += pad[1];
        u8o[2] += pad[2];
      }
    }
  }

  return groupDetections(all, minNeighbors);
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
