import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector } from './setup';

describe('Nonparametric Tests', () => {
  let wasm: any;

  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  describe('mannWhitneyU', () => {
    it('should detect difference between groups', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([6, 7, 8, 9, 10]);

      const result = wasm.mannWhitneyU(
        x,
        y,
        wasm.JsAlternative.TwoSided,
        true,  // continuity correction
        false, // not exact
        null,  // no conf level
        0.0    // mu
      );

      expect(result.statistic).toBeDefined();
      expect(result.p_value).toBeLessThan(0.05); // Groups are clearly different
    });

    it('should not detect difference for similar groups', async () => {
      const x = vector([1, 3, 5, 7, 9]);
      const y = vector([2, 4, 6, 8, 10]);

      const result = wasm.mannWhitneyU(
        x,
        y,
        wasm.JsAlternative.TwoSided,
        true,
        false,
        null,
        0.0
      );

      expect(result.p_value).toBeGreaterThan(0.05);
    });
  });

  describe('wilcoxonSignedRank', () => {
    it('should detect paired difference', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([2, 3, 4, 5, 6]); // Consistently higher

      const result = wasm.wilcoxonSignedRank(
        x,
        y,
        wasm.JsAlternative.TwoSided,
        true,
        false,
        null,
        0.0
      );

      expect(result.statistic).toBeDefined();
      expect(result.p_value).toBeLessThan(0.1);
    });
  });

  describe('kruskalWallis', () => {
    it('should detect difference among groups', async () => {
      const group1 = vector([1, 2, 3, 4, 5]);
      const group2 = vector([6, 7, 8, 9, 10]);
      const group3 = vector([11, 12, 13, 14, 15]);

      const result = wasm.kruskalWallis([group1, group2, group3]);

      expect(result.statistic).toBeGreaterThan(0);
      expect(result.p_value).toBeLessThan(0.01);
    });
  });
});
