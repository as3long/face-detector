export interface RawImage {
  data: Uint8Array;
  width: number;
  height: number;
}

export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  neighbor: number;
}

export interface CascadeFeature {
  size: number;
  px: number[];
  py: number[];
  pz: number[];
  nx: number[];
  ny: number[];
  nz: number[];
}

export interface CascadeStage {
  count: number;
  threshold: number;
  alpha: number[];
  feature: CascadeFeature[];
}

export interface CascadeData {
  count: number;
  width: number;
  height: number;
  stage_classifier: CascadeStage[];
}

export interface PrecompFeature {
  size: number;
  px: number[];
  pz: number[];
  nx: number[];
  nz: number[];
}

export interface PrecompStage {
  threshold: number;
  count: number;
  alpha: number[];
  features: PrecompFeature[];
}