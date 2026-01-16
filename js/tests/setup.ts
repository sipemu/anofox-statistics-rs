import { readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Path to the WASM package
const WASM_PKG_PATH = join(__dirname, '..', '..', 'crates', 'anofox-statistics-js', 'pkg');

let initialized = false;

/**
 * Initialize the WASM module for testing.
 */
export async function initWasm() {
  if (initialized) return;

  const wasmPath = join(WASM_PKG_PATH, 'anofox_statistics_js_bg.wasm');
  const wasmBuffer = await readFile(wasmPath);

  const wasmModule = await import(join(WASM_PKG_PATH, 'anofox_statistics_js.js'));
  wasmModule.initSync({ module: wasmBuffer });

  initialized = true;
}

/**
 * Get the WASM module exports.
 */
export async function getWasmModule() {
  await initWasm();
  return import(join(WASM_PKG_PATH, 'anofox_statistics_js.js'));
}

/**
 * Helper to create a Float64Array from a 1D array.
 */
export function vector(data: number[]): Float64Array {
  return new Float64Array(data);
}

/**
 * Assert that two numbers are approximately equal.
 */
export function expectClose(actual: number, expected: number, tolerance = 1e-6): void {
  const diff = Math.abs(actual - expected);
  if (diff > tolerance) {
    throw new Error(`Expected ${actual} to be close to ${expected} (diff: ${diff}, tolerance: ${tolerance})`);
  }
}

/**
 * Assert that two arrays are approximately equal.
 */
export function expectArrayClose(actual: ArrayLike<number>, expected: number[], tolerance = 1e-6): void {
  if (actual.length !== expected.length) {
    throw new Error(`Array length mismatch: ${actual.length} vs ${expected.length}`);
  }
  for (let i = 0; i < actual.length; i++) {
    expectClose(actual[i], expected[i], tolerance);
  }
}
