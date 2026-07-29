/**
 * BlazeFace - Lightweight Face Detection
 *
 * A TypeScript implementation of Google's BlazeFace algorithm.
 * "BlazeFace: Sub-millisecond Neural Face Detection on Mobiles"
 * (Bazarevsky et al., 2019)
 *
 * Architecture:
 * - 128×128 input, MobileNet-based feature extractor
 * - Two-scale detection: 16×16 + 8×8 feature maps
 * - 640 anchor boxes
 * - Outputs: bounding boxes + 6 facial landmarks
 *
 * @example
 * ```ts
 * import { BlazeFaceDetector } from './blazeface/index.js';
 *
 * const detector = new BlazeFaceDetector({ minConfidence: 0.75 });
 *
 * // From pre-computed model output
 * const faces = detector.detectFromModelOutput(scores, boxes, landmarks, imgW, imgH);
 *
 * // From raw image data (requires external model inference)
 * const inputTensor = detector.prepareInput(rgbaData, imgW, imgH);
 * // ... run model inference with inputTensor ...
 * const faces = await detector.detectFromPixels(rgbaData, imgW, imgH, { scores, boxes, landmarks });
 * ```
 */

export { BlazeFaceDetector } from './detector.js';
export type { BlazeFaceConfig, FaceDetection, Anchor } from './types.js';
export { generateAnchors, getAnchors, TOTAL_ANCHORS } from './anchors.js';
export { decodeDetections, nms } from './postprocess.js';
export { letterboxResize, hwcToChw, rgbaToRgb } from './preprocess.js';
