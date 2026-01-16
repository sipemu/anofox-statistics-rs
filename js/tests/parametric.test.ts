import { describe, it, expect, beforeAll } from 'vitest';
import { getWasmModule, vector, expectClose } from './setup';

describe('Parametric Tests', () => {
  let wasm: any;

  beforeAll(async () => {
    wasm = await getWasmModule();
  });

  describe('tTest', () => {
    it('should perform Welch t-test', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([2, 3, 4, 5, 6]);

      const result = wasm.tTest(
        x,
        y,
        wasm.JsTTestKind.Welch,
        wasm.JsAlternative.TwoSided,
        0.0,
        0.95
      );

      expect(result.statistic).toBeDefined();
      expect(result.p_value).toBeGreaterThan(0);
      expect(result.p_value).toBeLessThan(1);
      expect(result.df).toBeGreaterThan(0);
      expectClose(result.mean_x, 3.0);
      expectClose(result.mean_y, 4.0);
    });

    it('should perform Student t-test', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([2, 3, 4, 5, 6]);

      const result = wasm.tTest(
        x,
        y,
        wasm.JsTTestKind.Student,
        wasm.JsAlternative.TwoSided,
        0.0,
        0.95
      );

      expect(result.statistic).toBeDefined();
      expect(result.p_value).toBeGreaterThan(0);
    });

    it('should perform paired t-test', async () => {
      const x = vector([10, 12, 14, 16, 18, 20, 22]);
      const y = vector([11, 13, 13, 17, 19, 21, 24]);

      const result = wasm.tTest(
        x,
        y,
        wasm.JsTTestKind.Paired,
        wasm.JsAlternative.TwoSided,
        0.0,
        0.95
      );

      expect(result.statistic).toBeDefined();
      expect(result.p_value).toBeGreaterThanOrEqual(0);
      expect(result.p_value).toBeLessThanOrEqual(1);
    });

    it('should handle one-sided alternatives', async () => {
      const x = vector([1, 2, 3, 4, 5]);
      const y = vector([5, 6, 7, 8, 9]);

      const resultLess = wasm.tTest(
        x,
        y,
        wasm.JsTTestKind.Welch,
        wasm.JsAlternative.Less,
        0.0,
        null
      );

      const resultGreater = wasm.tTest(
        x,
        y,
        wasm.JsTTestKind.Welch,
        wasm.JsAlternative.Greater,
        0.0,
        null
      );

      // x < y, so "less" should have small p-value
      expect(resultLess.p_value).toBeLessThan(0.05);
      // x < y, so "greater" should have large p-value
      expect(resultGreater.p_value).toBeGreaterThan(0.5);
    });
  });

  describe('oneWayAnova', () => {
    it('should perform Fisher ANOVA', async () => {
      const group1 = vector([23, 25, 28, 31, 27]);
      const group2 = vector([31, 33, 35, 37, 34]);
      const group3 = vector([41, 43, 45, 47, 44]);

      const result = wasm.oneWayAnova(
        [group1, group2, group3],
        wasm.JsAnovaKind.Fisher
      );

      expect(result.statistic).toBeGreaterThan(0);
      expect(result.p_value).toBeLessThan(0.001); // Groups are clearly different
      expect(result.df_between).toBe(2);
      expect(result.n_groups).toBe(3);
    });

    it('should perform Welch ANOVA', async () => {
      const group1 = vector([23, 25, 28, 31, 27]);
      const group2 = vector([31, 33, 35, 37, 34]);
      const group3 = vector([41, 43, 45, 47, 44]);

      const result = wasm.oneWayAnova(
        [group1, group2, group3],
        wasm.JsAnovaKind.Welch
      );

      expect(result.statistic).toBeGreaterThan(0);
      expect(result.p_value).toBeLessThan(0.001);
    });
  });
});
