package com.sih26168.dr.engine

import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/**
 * Cross-language golden vectors (Window-Path Hardening P3).
 *
 * Reads the SAME `window_golden.json` asset Python's
 * `tests/test_window_golden.py` checks (produced by
 * `tools/gen_golden_vectors.py`), recomputes the window path in Kotlin
 * — gravity low-pass -> linear acc -> 10->100Hz resample -> normalize —
 * and asserts parity within 1e-6. Any drift between the training path
 * (Python) and the live path (Kotlin) fails here, not in the field.
 *
 * JVM unit test (no device): reads the asset from the module/repo layout.
 */
class WindowGoldenTest {

    private fun loadGolden(): JSONObject {
        val candidates = listOf(
            File("src/main/assets/window_golden.json"),             // cwd = android/app
            File("app/src/main/assets/window_golden.json"),          // cwd = android
            File("android/app/src/main/assets/window_golden.json"), // cwd = repo root
        )
        val f = candidates.firstOrNull { it.exists() }
            ?: throw IllegalStateException(
                "window_golden.json not found (looked in ${candidates.map { it.path }}) — " +
                    "run tools/gen_golden_vectors.py"
            )
        return JSONObject(f.readText())
    }

    private fun JSONObject.vec(key: String): List<DoubleArray> {
        val arr = getJSONArray(key)
        return List(arr.length()) { i ->
            val row = arr.getJSONArray(i)
            DoubleArray(row.length()) { j -> row.getDouble(j) }
        }
    }

    private fun org.json.JSONArray.toDoubleArray(): DoubleArray =
        DoubleArray(length()) { i -> getDouble(i) }

    /** Recompute the full path; returns (maxDiffVsSnapshot, meanAbsPhiDeg). */
    private fun recompute(): Pair<Double, Double> {
        val root = loadGolden()

        // Spec guard rides along: stale golden asset = stale APK contract.
        val specVersion = root.getInt("spec_version")
        assertTrue(
            "golden spec_version=$specVersion != APK ${WindowSpecGuard.EXPECTED_SPEC_VERSION}",
            specVersion == WindowSpecGuard.EXPECTED_SPEC_VERSION
        )

        // acc/gyro live under "fixture" (nested format from gen_golden_vectors.py)
        val fixture = root.getJSONObject("fixture")
        val acc = fixture.vec("acc_raw")
        val gyro = fixture.vec("gyro")
        val mean = root.getJSONObject("scaler").getJSONArray("mean").toDoubleArray()
        val std = root.getJSONObject("scaler").getJSONArray("std").toDoubleArray()
        val snap = root.vec("window_100hz_normalized")

        // 1) gravity low-pass on the 10Hz fixture — mirrors python
        //    core.signal.estimate_gravity_lowpass (alpha=0.02, init [0,0,9.81])
        //    and LeanDetector.gEst. ONE filter, same order as Python.
        val alpha = 0.02
        val g = doubleArrayOf(0.0, 0.0, 9.81)
        val lin = Array(acc.size) { DoubleArray(3) }
        var phiAbsSum = 0.0
        for (i in acc.indices) {
            for (c in 0..2) g[c] += alpha * (acc[i][c] - g[c])
            phiAbsSum += kotlin.math.abs(kotlin.math.atan2(g[1], g[2]))
            for (c in 0..2) lin[i][c] = acc[i][c] - g[c]
        }
        val meanAbsPhiDeg = Math.toDegrees(phiAbsSum / acc.size)

        // 2) resample 10Hz -> 100Hz: t_new = t0 + j*10ms, source step = 100ms.
        //    Linear interp of LINEAR acc + gyro (Python resamples AFTER
        //    gravity removal). i = j/10, frac = (j%10)/10.
        val n = snap.size
        var maxDiff = 0.0
        for (j in 0 until n) {
            val i = j / 10
            val frac = (j % 10) / 10.0
            val iN = if (i + 1 < acc.size) i + 1 else i
            for (c in 0..2) {
                val v = lin[i][c] + frac * (lin[iN][c] - lin[i][c])
                val norm = (v - mean[c]) / std[c]
                maxDiff = maxOf(maxDiff, kotlin.math.abs(norm - snap[j][c]))
            }
            for (c in 3..5) {
                val v = gyro[i][c - 3] + frac * (gyro[iN][c - 3] - gyro[i][c - 3])
                val norm = (v - mean[c]) / std[c]
                maxDiff = maxOf(maxDiff, kotlin.math.abs(norm - snap[j][c]))
            }
        }
        return Pair(maxDiff, meanAbsPhiDeg)
    }

    @Test
    fun windowPathMatchesPythonSnapshot() {
        val (maxDiff, _) = recompute()
        assertTrue(
            "window path parity FAILED: maxDiff=$maxDiff > 1e-6 — " +
                "Kotlin live path drifted from the Python training path",
            maxDiff <= 1e-6
        )
    }

    @Test
    fun leanTripwireUnder10deg() {
        val (_, phiDeg) = recompute()
        val limit = loadGolden().getDouble("phi_limit_deg")
        assertTrue(
            "lean tripwire: mean|phi|=$phiDeg deg >= limit $limit (38-deg bug regression?)",
            phiDeg < limit
        )
    }
}
