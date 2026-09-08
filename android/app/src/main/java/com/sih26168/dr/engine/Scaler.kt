package com.sih26168.dr.engine

import android.content.Context
import org.json.JSONObject

/**
 * Per-channel mean/std normalization — mirrors python/scaler.json (train-only stats).
 *
 * Verifies the window-spec fingerprint at construction (P2): a scaler built
 * against a different window path than this APK's spec refuses to load.
 *
 * @param context used to open `assets/scaler.json` (copied from repo root at build).
 * @throws WindowSpecGuard.SpecMismatch when the fingerprint is absent or stale.
 */
class Scaler(context: Context) {
    private val mean = FloatArray(6)
    private val std = FloatArray(6)

    /** Spec version carried by the bundled scaler.json (for the health sheet). */
    val specVersion: Int

    init {
        WindowSpecGuard.verifyScaler(context)
        val json = context.assets.open("scaler.json").bufferedReader().use { it.readText() }
        val obj = JSONObject(json)
        specVersion = obj.optInt("spec_version", -1)
        val m = obj.getJSONArray("mean")
        val s = obj.getJSONArray("std")
        for (i in 0 until 6) {
            mean[i] = m.getDouble(i).toFloat()
            std[i] = s.getDouble(i).toFloat()
        }
        // Degenerate std (0 or NaN) would silently zero/garbage every window.
        for (i in 0 until 6) {
            if (std[i].isNaN() || std[i] <= 0f) {
                throw IllegalStateException("scaler.json std[$i]=${std[i]} — refusing to normalize")
            }
        }
    }

    /**
     * Normalize one raw 6-channel sample in place into [out].
     *
     * @param raw 6 channels `[linAcc(3), gyro(3)]` in physical units.
     * @param out destination array of size 6 (reused buffer).
     */
    fun normalize(raw: FloatArray, out: FloatArray) {
        for (i in 0 until 6) out[i] = (raw[i] - mean[i]) / std[i]
    }
}
