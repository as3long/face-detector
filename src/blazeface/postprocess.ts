/**
 * BlazeFace Post-Processing
 *
 * Decodes raw model outputs into face detections with:
 * - Anchor-based box decoding
 * - Sigmoid score activation
 * - Non-Maximum Suppression with tie-breaking
 */

import type { Anchor, FaceDetection, ProcessedDetection, RawDetection } from './types.js';
import { getAnchors } from './anchors.js';

/** Sigmoid activation function */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** Compute IoU between two boxes */
function iou(
  a: { x1: number; y1: number; x2: number; y2: number },
  b: { x1: number; y1: number; x2: number; y2: number },
): number {
  const interX1 = Math.max(a.x1, b.x1);
  const interY1 = Math.max(a.y1, b.y1);
  const interX2 = Math.min(a.x2, b.x2);
  const interY2 = Math.min(a.y2, b.y2);

  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const interArea = interW * interH;

  const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
  const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
  const unionArea = aArea + bArea - interArea;

  return unionArea > 0 ? interArea / unionArea : 0;
}

/**
 * Decode raw model outputs into processed detections.
 *
 * @param scores - Raw score logits [numAnchors]
 * @param boxes - Raw box regression [numAnchors × 4]
 * @param landmarks - Raw landmark regression [numAnchors × 12]
 * @param anchors - Pre-computed anchor boxes
 * @param minConfidence - Minimum confidence threshold
 * @returns Array of processed detections in normalized coordinates
 */
export function decodeDetections(
  scores: Float32Array,
  boxes: Float32Array,
  landmarks: Float32Array,
  anchors: Anchor[],
  minConfidence: number,
): ProcessedDetection[] {
  const detections: ProcessedDetection[] = [];
  const numAnchors = anchors.length;

  // BlazeFace uses these variance values for box decoding
  const boxVariance: [number, number, number, number] = [0.1, 0.1, 0.2, 0.2];
  const landmarkVariance = 0.1;

  for (let i = 0; i < numAnchors; i++) {
    const score = sigmoid(scores[i]);
    if (score < minConfidence) continue;

    const anchor = anchors[i];
    const baseIdx = i * 4;

    // Decode box: center offset + size
    const dx = boxes[baseIdx] * boxVariance[0];
    const dy = boxes[baseIdx + 1] * boxVariance[1];
    const dw = boxes[baseIdx + 2] * boxVariance[2];
    const dh = boxes[baseIdx + 3] * boxVariance[3];

    const cx = anchor.cx + dx * anchor.w;
    const cy = anchor.cy + dy * anchor.h;
    const w = anchor.w * Math.exp(dw);
    const h = anchor.h * Math.exp(dh);

    // Convert to corner coordinates (normalized)
    const x1 = cx - w / 2;
    const y1 = cy - h / 2;
    const x2 = cx + w / 2;
    const y2 = cy + h / 2;

    // Decode landmarks
    const lmBaseIdx = i * 12;
    const decodedLandmarks: number[] = [];
    for (let j = 0; j < 6; j++) {
      const lx = anchor.cx + landmarks[lmBaseIdx + j * 2] * landmarkVariance * anchor.w;
      const ly = anchor.cy + landmarks[lmBaseIdx + j * 2 + 1] * landmarkVariance * anchor.h;
      decodedLandmarks.push(lx, ly);
    }

    detections.push({
      x1: Math.max(0, x1),
      y1: Math.max(0, y1),
      x2: Math.min(1, x2),
      y2: Math.min(1, y2),
      score,
      landmarks: decodedLandmarks,
      anchorIdx: i,
    });
  }

  return detections;
}

/**
 * Non-Maximum Suppression with tie-breaking.
 * BlazeFace uses a specific NMS variant that handles tied scores.
 *
 * @param detections - Decoded detections
 * @param iouThreshold - IoU threshold for suppression
 * @param maxDetections - Maximum number of detections to return
 * @returns Filtered detections
 */
export function nms(
  detections: ProcessedDetection[],
  iouThreshold: number,
  maxDetections: number,
): ProcessedDetection[] {
  if (detections.length === 0) return [];

  // Sort by score descending
  const sorted = [...detections].sort((a, b) => b.score - a.score);

  const selected: ProcessedDetection[] = [];
  const suppressed = new Set<number>();

  for (let i = 0; i < sorted.length && selected.length < maxDetections; i++) {
    if (suppressed.has(i)) continue;

    selected.push(sorted[i]);

    // Suppress overlapping detections
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue;
      const overlap = iou(sorted[i], sorted[j]);
      if (overlap > iouThreshold) {
        suppressed.add(j);
      }
    }
  }

  return selected;
}

/**
 * Convert processed detections to FaceDetection format.
 *
 * @param detections - NMS-filtered detections
 * @param inputWidth - Model input width
 * @param inputHeight - Model input height
 * @param imageWidth - Original image width
 * @param imageHeight - Original image height
 * @param offsetX - Letterbox offset x (for coordinate mapping)
 * @param offsetY - Letterbox offset y
 * @param scaleX - Scale from input to image space
 * @param scaleY - Scale from input to image space
 */
export function toFaceDetections(
  detections: ProcessedDetection[],
  inputWidth: number,
  inputHeight: number,
  imageWidth: number,
  imageHeight: number,
  offsetX: number,
  offsetY: number,
  scaleX: number,
  scaleY: number,
): FaceDetection[] {
  return detections.map((det) => {
    // Map from normalized [0,1] to image pixel coordinates
    const x = (det.x1 * inputWidth - offsetX) / imageWidth;
    const y = (det.y1 * inputHeight - offsetY) / imageHeight;
    const w = ((det.x2 - det.x1) * inputWidth * scaleX) / imageWidth;
    const h = ((det.y2 - det.y1) * inputHeight * scaleY) / imageHeight;

    const landmarks: [number, number, number, number, number, number, number, number, number, number, number, number] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let j = 0; j < 6; j++) {
      landmarks[j * 2] = (det.landmarks[j * 2] * inputWidth - offsetX) / imageWidth;
      landmarks[j * 2 + 1] = (det.landmarks[j * 2 + 1] * inputHeight - offsetY) / imageHeight;
    }

    return {
      x: Math.max(0, Math.min(1, x)),
      y: Math.max(0, Math.min(1, y)),
      width: Math.max(0, Math.min(1 - x, w)),
      height: Math.max(0, Math.min(1 - y, h)),
      confidence: det.score,
      landmarks,
    };
  });
}
