/**
 * BlazeFace type definitions
 * Based on: "BlazeFace: Sub-millisecond Neural Face Detection on Mobiles" (Bazarevsky et al., 2019)
 */

/** Detection result for a single face */
export interface FaceDetection {
  /** Bounding box top-left x (normalized 0-1) */
  x: number;
  /** Bounding box top-left y (normalized 0-1) */
  y: number;
  /** Bounding box width (normalized 0-1) */
  width: number;
  /** Bounding box height (normalized 0-1) */
  height: number;
  /** Detection confidence score [0, 1] */
  confidence: number;
  /** 6 facial landmarks (normalized 0-1): [rightEye, leftEye, noseTip, mouthCenter, rightEarTragion, leftEarTragion] */
  landmarks: [number, number, number, number, number, number, number, number, number, number, number, number];
}

/** Raw anchor box definition */
export interface Anchor {
  /** Center x (normalized) */
  cx: number;
  /** Center y (normalized) */
  cy: number;
  /** Width (normalized) */
  w: number;
  /** Height (normalized) */
  h: number;
}

/** Raw model output before post-processing */
export interface RawDetection {
  /** Anchor index */
  anchorIdx: number;
  /** Class score (before sigmoid) */
  score: number;
  /** Box regression: [dx, dy, dw, dh] */
  boxRegression: [number, number, number, number];
  /** Landmark regression: 12 values */
  landmarkRegression: number[];
}

/** BlazeFace detector configuration */
export interface BlazeFaceConfig {
  /** Model input width (default: 128) */
  inputWidth?: number;
  /** Model input height (default: 128) */
  inputHeight?: number;
  /** Minimum confidence threshold (default: 0.75) */
  minConfidence?: number;
  /** NMS IoU threshold (default: 0.3) */
  nmsIouThreshold?: number;
  /** Maximum number of detections (default: 100) */
  maxDetections?: number;
}

/** Internal: processed detection for NMS */
export interface ProcessedDetection {
  /** Bounding box in pixel coordinates */
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** Confidence score */
  score: number;
  /** Landmarks in pixel coordinates */
  landmarks: number[];
  /** Original anchor index */
  anchorIdx: number;
}
