/** @type {import('jest').Config} */
module.exports = {
  preset: 'ts-jest/presets/default-esm',
  testEnvironment: 'node',
  roots: ['<rootDir>/test'],
  moduleNameMapper: {
    '^(\\.{1,2}/.*)\\.js$': '$1',
  },
  transform: {
    '^.+\\.tsx?$': [
      'ts-jest',
      {
        tsconfig: {
          esModuleInterop: true,
          resolveJsonModule: true,
          target: 'ES2022',
          module: 'commonjs',
          moduleResolution: 'node',
          allowJs: true,
          skipLibCheck: true,
        },
      },
    ],
  },
};
