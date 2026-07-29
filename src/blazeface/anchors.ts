/**
 * BlazeFace Anchor Generator
 *
 * Generates anchor boxes for the two feature map scales used by BlazeFace:
 * - 16×16 feature map (stride 8 from 128×128 input)
 * - 8×8 feature map (stride 16 from 128×128 input)
 *
 * Each feature map cell produces 2 anchors with different sizes.
 * Total anchors = 16×16×2 + 8×8×2 = 512 + 128 = 640
 */

import type { Anchor } from './types.js';

/**
 * Generate all anchors for BlazeFace.
 * Anchors are in normalized coordinates [0, 1].
 */
export function generateAnchors(
  inputWidth: number = 128,
  inputHeight: number = 128,
): Anchor[] {
  const anchors: Anchor[] = [];

  // Feature map 1: 16×16 (stride 8)
  // Anchor sizes: ~0.0625 (8px) and ~0.125 (16px) of image
  const stride1 = 8;
  const fm1Width = Math.floor(inputWidth / stride1);   // 16
  const fm1Height = Math.floor(inputHeight / stride1);  // 16
  const sizes1 = [stride1 / inputWidth, (stride1 * 2) / inputWidth]; // [0.0625, 0.125]

  for (let y = 0; y < fm1Height; y++) {
    for (let x = 0; x < fm1Width; x++) {
      const cx = (x + 0.5) / fm1Width;
      const cy = (y + 0.5) / fm1Height;
      for (const size of sizes1) {
        anchors.push({ cx, cy, w: size, h: size });
      }
    }
  }

  // Feature map 2: 8×8 (stride 16)
  // Anchor sizes: ~0.25 (32px) and ~0.5 (64px) of image
  const stride2 = 16;
  const fm2Width = Math.floor(inputWidth / stride2);   // 8
  const fm2Height = Math.floor(inputHeight / stride2);  // 8
  const sizes2 = [stride2 / inputWidth, (stride2 * 2) / inputWidth]; // [0.125, 0.25]

  for (let y = 0; y < fm2Height; y++) {
    for (let x = 0; x < fm2Width; x++) {
      const cx = (x + 0.5) / fm2Width;
      const cy = (y + 0.5) / fm2Height;
      for (const size of sizes2) {
        anchors.push({ cx, cy, w: size, h: size });
      }
    }
  }

  return anchors;
}

/** Pre-computed anchor cache */
let cachedAnchors: Anchor[] | null = null;
let cachedInputSize = 0;

/**
 * Get anchors for the given input size, using cache.
 */
export function getAnchors(inputWidth: number, inputHeight: number): Anchor[] {
  const size = inputWidth * 1000 + inputHeight;
  if (cachedAnchors && cachedInputSize === size) {
    return cachedAnchors;
  }
  cachedAnchors = generateAnchors(inputWidth, inputHeight);
  cachedInputSize = size;
  return cachedAnchors;
}

/** Total number of anchors for a 128×128 input */
export const TOTAL_ANCHORS = 640; // 16*16*2 + 8*8*2
