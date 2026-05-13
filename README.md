# face-detector

A face detection library based on the Viola-Jones algorithm. Detects faces in images using a cascade of Haar-like features.

## Installation

```bash
npm install face-detector
```

## Usage

### ESM

```typescript
import { detectFaces, detectBestFace } from 'face-detector';

// Detect all faces in an image
const faces = await detectFaces('path/to/image.jpg');
console.log(faces);
// [
//   { x: 100, y: 50, width: 200, height: 200, confidence: 0.82, neighbor: 5 },
//   ...
// ]

// With performance/accuracy options
const faces2 = await detectFaces('path/to/image.jpg', {
  maxDimension: 320,         // downscale large images (default 500)
  minNeighbors: 4,           // require more overlap (default 3)
  scaleFactor: 1.1,          // finer pyramid scale (default 1.19)
  step: 2,                   // smaller stride = more scans (default 4)
});

// Get the best (largest) face
const best = await detectBestFace('path/to/image.jpg');
console.log(best);
// { x: 100, y: 50, width: 200, height: 200 }
```

### CommonJS

```javascript
const { detectFaces, detectBestFace } = require('face-detector');

async function main() {
  const faces = await detectFaces('path/to/image.jpg');
  console.log(faces);
}
```

## API

### `detectFaces(imagePath: string, options?: DetectOptions): Promise<Detection[]>`

Detects all faces in the given image. Returns an array of detection results.

- `imagePath` - Path to the image file (supported formats: JPEG, PNG, WebP, etc.)
- `options` - Optional configuration object

**DetectOptions**:
| Field          | Type     | Default                     | Description                                      |
|----------------|----------|-----------------------------|--------------------------------------------------|
| `maxDimension` | `number` | `200`                       | Max image dimension before downscaling (max 500) |
| `minNeighbors` | `number` | `3`                         | Minimum overlapping detections to keep a face    |
| `scaleFactor`  | `number` | `1.19` (2^(1/4))            | Scale multiplier between pyramid levels          |
| `step`         | `number` | `4`                         | Sliding window stride (smaller = more scans)     |

**Detection**:
| Field       | Type     | Description                          |
|-------------|----------|--------------------------------------|
| `x`         | `number` | X coordinate of the face             |
| `y`         | `number` | Y coordinate of the face             |
| `width`     | `number` | Width of the face bounding box       |
| `height`    | `number` | Height of the face bounding box      |
| `confidence`| `number` | Detection confidence (0-1)           |
| `neighbor`  | `number` | Number of overlapping detections     |

### `detectBestFace(imagePath: string, options?: DetectOptions): Promise<{ x, y, width, height } | null>`

Detects the largest (most prominent) face in the image. Returns `null` if no face is found.

## Dependencies

- [sharp](https://sharp.pixelplumbing.com/) - High-performance image processing

## License

MIT
