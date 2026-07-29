/**
 * BlazeFace Image Preprocessing
 *
 * Handles:
 * - Letterbox resize to model input size
 * - Float32 normalization (0-255 → -1 to 1 or 0 to 1)
 * - RGB channel reordering
 */

/** Result of letterbox resize with offset information */
export interface LetterboxResult {
  /** Resized pixel data as Float32Array, normalized, RGB order */
  data: Float32Array;
  /** Width of the resized image */
  width: number;
  /** Height of the resized image */
  height: number;
  /** X offset of the image within the letterbox */
  offsetX: number;
  /** Y offset of the image within the letterbox */
  offsetY: number;
  /** Scale factor applied */
  scale: number;
}

/**
 * Letterbox resize an image to the target size while preserving aspect ratio.
 * Pads with gray (128, 128, 128) to fill the target dimensions.
 *
 * @param imageData - Source RGBA pixel data (from canvas.getImageData or sharp)
 * @param srcWidth - Source image width
 * @param srcHeight - Source image height
 * @param targetWidth - Target width (e.g., 128)
 * @param targetHeight - Target height (e.g., 128)
 * @returns Normalized float32 data in RGB order, plus offset info
 */
export function letterboxResize(
  imageData: Uint8Array | Uint8ClampedArray,
  srcWidth: number,
  srcHeight: number,
  targetWidth: number,
  targetHeight: number,
): LetterboxResult {
  const output = new Float32Array(targetWidth * targetHeight * 3);
  const padValue = 128 / 255; // normalized gray

  // Fill with pad color
  for (let i = 0; i < output.length; i += 3) {
    output[i] = padValue;
    output[i + 1] = padValue;
    output[i + 2] = padValue;
  }

  // Calculate scale to fit
  const scaleX = targetWidth / srcWidth;
  const scaleY = targetHeight / srcHeight;
  const scale = Math.min(scaleX, scaleY);

  const newWidth = Math.round(srcWidth * scale);
  const newHeight = Math.round(srcHeight * scale);
  const offsetX = Math.floor((targetWidth - newWidth) / 2);
  const offsetY = Math.floor((targetHeight - newHeight) / 2);

  // Bilinear interpolation resize
  for (let dy = 0; dy < newHeight; dy++) {
    for (let dx = 0; dx < newWidth; dx++) {
      const srcX = (dx / newWidth) * srcWidth;
      const srcY = (dy / newHeight) * srcHeight;

      const x0 = Math.floor(srcX);
      const y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, srcWidth - 1);
      const y1 = Math.min(y0 + 1, srcHeight - 1);

      const fx = srcX - x0;
      const fy = srcY - y0;

      // Sample 4 corners
      const idx00 = (y0 * srcWidth + x0) * 4;
      const idx10 = (y0 * srcWidth + x1) * 4;
      const idx01 = (y1 * srcWidth + x0) * 4;
      const idx11 = (y1 * srcWidth + x1) * 4;

      const outIdx = ((offsetY + dy) * targetWidth + (offsetX + dx)) * 3;

      for (let c = 0; c < 3; c++) {
        // Source is RGBA, we want RGB
        const v00 = imageData[idx00 + c] / 255;
        const v10 = imageData[idx10 + c] / 255;
        const v01 = imageData[idx01 + c] / 255;
        const v11 = imageData[idx11 + c] / 255;

        const v = v00 * (1 - fx) * (1 - fy)
                + v10 * fx * (1 - fy)
                + v01 * (1 - fx) * fy
                + v11 * fx * fy;

        output[outIdx + c] = v;
      }
    }
  }

  return {
    data: output,
    width: targetWidth,
    height: targetHeight,
    offsetX,
    offsetY,
    scale,
  };
}

/**
 * Convert HWC float32 data to CHW format (needed for most model inputs).
 *
 * @param hwcData - Input data in H×W×C format
 * @param height - Image height
 * @param width - Image width
 * @param channels - Number of channels (3 for RGB)
 * @returns Data in C×H×W format
 */
export function hwcToChw(
  hwcData: Float32Array,
  height: number,
  width: number,
  channels: number = 3,
): Float32Array {
  const size = height * width;
  const chwData = new Float32Array(channels * size);

  for (let c = 0; c < channels; c++) {
    for (let i = 0; i < size; i++) {
      chwData[c * size + i] = hwcData[i * channels + c];
    }
  }

  return chwData;
}

/**
 * Convert RGBA image data to RGB.
 *
 * @param rgba - RGBA pixel data
 * @param width - Image width
 * @param height - Image height
 * @returns RGB pixel data
 */
export function rgbaToRgb(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
): Uint8Array {
  const pixelCount = width * height;
  const rgb = new Uint8Array(pixelCount * 3);

  for (let i = 0; i < pixelCount; i++) {
    rgb[i * 3] = rgba[i * 4];
    rgb[i * 3 + 1] = rgba[i * 4 + 1];
    rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }

  return rgb;
}
