/**
 * BlazeFace Detector
 *
 * A lightweight face detection model based on:
 * "BlazeFace: Sub-millisecond Neural Face Detection on Mobiles"
 * (Bazarevsky et al., 2019)
 *
 * Architecture:
 * - MobileNetV1/V2-based feature extractor with depthwise separable convolutions
 * - Two-scale feature maps: 16×16 and 8×8 (from 128×128 input)
 * - Single-shot detection with 640 anchor boxes
 * - Outputs: bounding boxes + 6 facial landmarks (2 eyes, nose, mouth, 2 ears)
 *
 * This implementation supports:
 * - Node.js (via sharp for image loading)
 * - Browser (via Canvas API)
 * - Any ONNX-compatible runtime for inference
 */

import type { Anchor, BlazeFaceConfig, FaceDetection, ProcessedDetection } from './types.js';
import { getAnchors, TOTAL_ANCHORS } from './anchors.js';
import { decodeDetections, nms, toFaceDetections } from './postprocess.js';
import { letterboxResize, hwcToChw } from './preprocess.js';

/** Default BlazeFace configuration */
const DEFAULT_CONFIG: Required<BlazeFaceConfig> = {
  inputWidth: 128,
  inputHeight: 128,
  minConfidence: 0.75,
  nmsIouThreshold: 0.3,
  maxDetections: 100,
};

/**
 * BlazeFace face detector.
 *
 * Usage:
 * ```ts
 * const detector = new BlazeFaceDetector({ minConfidence: 0.5 });
 * const faces = await detector.detect(imageData, width, height);
 * ```
 */
export class BlazeFaceDetector {
  private config: Required<BlazeFaceConfig>;
  private anchors: Anchor[];

  constructor(config?: BlazeFaceConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.anchors = getAnchors(this.config.inputWidth, this.config.inputHeight);
  }

  /**
   * Run face detection on raw image data.
   *
   * @param imageData - RGBA pixel data (Uint8Array or Uint8ClampedArray)
   * @param imageWidth - Image width in pixels
   * @param imageHeight - Image height in pixels
   * @param modelOutput - Optional pre-computed model output (scores, boxes, landmarks)
   * @returns Array of face detections with normalized coordinates
   */
  async detectFromPixels(
    imageData: Uint8Array | Uint8ClampedArray,
    imageWidth: number,
    imageHeight: number,
    modelOutput?: { scores: Float32Array; boxes: Float32Array; landmarks: Float32Array },
  ): Promise<FaceDetection[]> {
    const { inputWidth, inputHeight, minConfidence, nmsIouThreshold, maxDetections } = this.config;

    // Preprocess: letterbox resize + normalize
    const preprocessed = letterboxResize(imageData, imageWidth, imageHeight, inputWidth, inputHeight);

    // If no model output provided, return empty (needs external inference)
    if (!modelOutput) {
      throw new Error(
        'BlazeFaceDetector.detectFromPixels requires modelOutput. ' +
        'Use detectFromModelOutput() if you have pre-computed model outputs, ' +
        'or use the ONNX-based detector.'
      );
    }

    // Decode detections from model output
    const rawDetections = decodeDetections(
      modelOutput.scores,
      modelOutput.boxes,
      modelOutput.landmarks,
      this.anchors,
      minConfidence,
    );

    // Apply NMS
    const filtered = nms(rawDetections, nmsIouThreshold, maxDetections);

    // Convert to normalized face detections
    return toFaceDetections(
      filtered,
      inputWidth,
      inputHeight,
      imageWidth,
      imageHeight,
      preprocessed.offsetX,
      preprocessed.offsetY,
      preprocessed.scale,
      preprocessed.scale,
    );
  }

  /**
   * Run face detection from pre-computed model output.
   * Use this when you have already run the model and have raw output tensors.
   *
   * @param scores - Raw score logits [numAnchors]
   * @param boxes - Raw box regression [numAnchors × 4]
   * @param landmarks - Raw landmark regression [numAnchors × 12]
   * @param imageWidth - Original image width
   * @param imageHeight - Original image height
   * @returns Array of face detections
   */
  detectFromModelOutput(
    scores: Float32Array,
    boxes: Float32Array,
    landmarks: Float32Array,
    imageWidth: number,
    imageHeight: number,
  ): FaceDetection[] {
    const { inputWidth, inputHeight, minConfidence, nmsIouThreshold, maxDetections } = this.config;

    const rawDetections = decodeDetections(scores, boxes, landmarks, this.anchors, minConfidence);
    const filtered = nms(rawDetections, nmsIouThreshold, maxDetections);

    // No letterbox offset — assume model output is already in input space
    return toFaceDetections(
      filtered,
      inputWidth,
      inputHeight,
      imageWidth,
      imageHeight,
      0, 0,
      1, 1,
    );
  }

  /**
   * Get model input dimensions.
   */
  getInputSize(): { width: number; height: number } {
    return { width: this.config.inputWidth, height: this.config.inputHeight };
  }

  /**
   * Get the number of anchors used by this model.
   */
  getAnchorCount(): number {
    return this.anchors.length;
  }

  /**
   * Prepare input tensor for model inference.
   * Converts RGBA image data to the CHW float32 format expected by the model.
   *
   * @param imageData - RGBA pixel data
   * @param imageWidth - Image width
   * @param imageHeight - Image height
   * @returns Float32Array in NCHW format [1, 3, H, W], normalized to [0, 1]
   */
  prepareInput(
    imageData: Uint8Array | Uint8ClampedArray,
    imageWidth: number,
    imageHeight: number,
  ): Float32Array {
    const { inputWidth, inputHeight } = this.config;
    const preprocessed = letterboxResize(imageData, imageWidth, imageHeight, inputWidth, inputHeight);
    return hwcToChw(preprocessed.data, inputHeight, inputWidth, 3);
  }
}

// Re-export types and utilities
export type { BlazeFaceConfig, FaceDetection, Anchor } from './types.js';
export { generateAnchors, getAnchors, TOTAL_ANCHORS } from './anchors.js';
export { decodeDetections, nms } from './postprocess.js';
export { letterboxResize, hwcToChw, rgbaToRgb } from './preprocess.js';
