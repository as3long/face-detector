import * as path from 'path';
import * as fs from 'fs';
import { detectFaces, detectBestFace } from '../src/detector.js';

const faceDir = path.resolve(__dirname, 'images', 'face');
const nofaceDir = path.resolve(__dirname, 'images', 'noface');

function getImages(dir: string): string[] {
  return fs.readdirSync(dir)
    .filter((f: string) => /\.(jpg|jpeg|png|webp)$/i.test(f))
    .map((f: string) => path.join(dir, f));
}

const faceImages = getImages(faceDir);
const nofaceImages = getImages(nofaceDir);

beforeAll(() => {
  expect(faceImages.length).toBeGreaterThan(0);
  expect(nofaceImages.length).toBeGreaterThan(0);
});

describe('detectFaces', () => {
  it('returns empty array for no-face images', async () => {
    const results = await Promise.all(
      nofaceImages.map(img => detectFaces(img))
    );
    for (const faces of results) {
      expect(faces).toEqual([]);
    }
  }, 60000);

  it('returns exactly one face for every face image', async () => {
    const results = await Promise.all(
      faceImages.map(img => detectFaces(img))
    );
    for (const faces of results) {
      expect(faces.length).toBe(1);
    }
  }, 60000);

  it('returns valid detection shape', async () => {
    const faces = await detectFaces(faceImages[0]);
    for (const f of faces) {
      expect(f).toHaveProperty('x');
      expect(f).toHaveProperty('y');
      expect(f).toHaveProperty('width');
      expect(f).toHaveProperty('height');
      expect(f).toHaveProperty('confidence');
      expect(f).toHaveProperty('neighbor');
      expect(typeof f.x).toBe('number');
      expect(typeof f.y).toBe('number');
      expect(typeof f.width).toBe('number');
      expect(typeof f.height).toBe('number');
      expect(typeof f.confidence).toBe('number');
      expect(typeof f.neighbor).toBe('number');
      expect(f.confidence).toBeGreaterThanOrEqual(0);
      expect(f.confidence).toBeLessThanOrEqual(1);
      expect(f.neighbor).toBeGreaterThanOrEqual(1);
    }
  }, 60000);

  it('accepts maxDimension option', async () => {
    const faces = await detectFaces(faceImages[0], { maxDimension: 200 });
    expect(Array.isArray(faces)).toBe(true);
  }, 60000);

  it('accepts minNeighbors option', async () => {
    const faces = await detectFaces(faceImages[0], { minNeighbors: 5 });
    expect(Array.isArray(faces)).toBe(true);
  }, 60000);

  it('accepts scaleFactor option', async () => {
    const faces = await detectFaces(faceImages[0], { scaleFactor: 1.1 });
    expect(Array.isArray(faces)).toBe(true);
  }, 60000);

  it('accepts step option', async () => {
    const faces = await detectFaces(faceImages[0], { step: 8 });
    expect(Array.isArray(faces)).toBe(true);
  }, 60000);

  it('accepts all options together', async () => {
    const faces = await detectFaces(faceImages[0], {
      maxDimension: 400,
      minNeighbors: 4,
      scaleFactor: 1.15,
      step: 4,
    });
    expect(Array.isArray(faces)).toBe(true);
  }, 60000);

  it('larger stride produces fewer or equal detections', async () => {
    const [tight, wide] = await Promise.all([
      detectFaces(faceImages[0], { step: 4 }),
      detectFaces(faceImages[0], { step: 8 }),
    ]);
    expect(tight.length).toBeGreaterThanOrEqual(wide.length);
  }, 60000);

  it('higher minNeighbors produces fewer or equal detections', async () => {
    const [loose, strict] = await Promise.all([
      detectFaces(faceImages[0], { minNeighbors: 1 }),
      detectFaces(faceImages[0], { minNeighbors: 5 }),
    ]);
    expect(loose.length).toBeGreaterThanOrEqual(strict.length);
  }, 60000);

  it('smaller maxDimension may change results', async () => {
    const [full, downscaled] = await Promise.all([
      detectFaces(faceImages[0], { maxDimension: 800 }),
      detectFaces(faceImages[0], { maxDimension: 100 }),
    ]);
    expect(Array.isArray(full)).toBe(true);
    expect(Array.isArray(downscaled)).toBe(true);
  }, 60000);
});


describe('detectBestFace', () => {
  it('returns valid shape for every face image', async () => {
    for (const img of faceImages) {
      const result = await detectBestFace(img);
      expect(result).not.toBeNull();
      expect(result).toHaveProperty('x');
      expect(result).toHaveProperty('y');
      expect(result).toHaveProperty('width');
      expect(result).toHaveProperty('height');
      expect(typeof result!.x).toBe('number');
      expect(typeof result!.y).toBe('number');
      expect(typeof result!.width).toBe('number');
      expect(typeof result!.height).toBe('number');
    }
  }, 60000);

  it('accepts options', async () => {
    const result = await detectBestFace(faceImages[0], { maxDimension: 300 });
    expect(result).not.toBeNull();
    expect(typeof result!.x).toBe('number');
  }, 60000);

  it('returns null for no-face images', async () => {
    for (const img of nofaceImages) {
      const result = await detectBestFace(img);
      expect(result).toBeNull();
    }
  }, 60000);
});
