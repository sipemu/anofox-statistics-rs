import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector, expectClose } from './setup';

describe('Correlation Tests', () => {
  let wasm: any;

  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  describe('pearsonCorrelation', () => {
    it('should detect perfect positive correlation', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([2, 4, 6, 8, 10]);

      const result = wasm.pearsonCorrelation(x, y, 0.95);

      expectClose(result.estimate, 1.0, 1e-10);
      expect(result.p_value).toBeLessThan(0.001);
    });

    it('should detect perfect negative correlation', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([10, 8, 6, 4, 2]);

      const result = wasm.pearsonCorrelation(x, y, 0.95);

      expectClose(result.estimate, -1.0, 1e-10);
      expect(result.p_value).toBeLessThan(0.001);
    });

    it('should detect no correlation', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([5, 2, 4, 1, 3]);

      const result = wasm.pearsonCorrelation(x, y, 0.95);

      expect(Math.abs(result.estimate)).toBeLessThan(0.5);
      expect(result.p_value).toBeGreaterThan(0.05);
    });
  });

  describe('spearmanCorrelation', () => {
    it('should detect monotonic relationship', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([1, 4, 9, 16, 25]); // y = x^2, monotonic but not linear

      const result = wasm.spearmanCorrelation(x, y, 0.95);

      expectClose(result.estimate, 1.0, 1e-10);
    });
  });

  describe('kendallCorrelation', () => {
    it('should compute Kendall tau-b', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([2, 4, 6, 8, 10]);

      const result = wasm.kendallCorrelation(x, y, wasm.JsKendallVariant.TauB);

      expectClose(result.estimate, 1.0, 1e-10);
    });
  });
});
