import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector, expectClose } from './setup';

describe('Distributional Tests', () => {
  let wasm: any;

  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  describe('shapiroWilk', () => {
    it('should pass for normal data', async () => {
      // Approximately normal data
      const data = vector([2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7, 3.0, 2.6, 3.3]);

      const result = wasm.shapiroWilk(data);

      expect(result.statistic).toBeGreaterThan(0);
      expect(result.statistic).toBeLessThanOrEqual(1);
      expect(result.p_value).toBeGreaterThan(0.05); // Should not reject normality
    });

    it('should fail for non-normal data', async () => {
      // Clearly non-normal (bimodal)
      const data = vector([1, 1, 1, 1, 1, 10, 10, 10, 10, 10]);

      const result = wasm.shapiroWilk(data);

      expect(result.statistic).toBeGreaterThan(0);
      expect(result.p_value).toBeLessThan(0.05); // Should reject normality
    });
  });

  describe('dagostinoKSquared', () => {
    it('should test for normality', async () => {
      // Need at least 20 samples for D'Agostino test
      const data = vector([
        2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7, 3.0, 2.6, 3.3,
        2.4, 3.0, 2.9, 3.1, 2.8, 3.2, 2.7, 3.0, 2.5, 3.4
      ]);

      const result = wasm.dagostinoKSquared(data);

      expect(result.k2_statistic).toBeGreaterThanOrEqual(0);
      expect(result.p_value).toBeGreaterThan(0);
      expect(result.skewness_z).toBeDefined();
      expect(result.kurtosis_z).toBeDefined();
    });
  });
});
