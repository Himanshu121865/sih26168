package com.sih26168.dr.engine

import android.content.Context
import org.json.JSONObject

/**
 * Per-channel mean/std normalization — mirrors python/scaler.json (train-only stats).
 *
 * @param context used to open `assets/scaler.json` (copied from repo root at build).
 */
class Scaler(context: Context) {
    private val mean = FloatArray(6)
    private val std = FloatArray(6)

    init {
        val json = context.assets.open("scaler.json").bufferedReader().use { it.readText() }
        val obj = JSONObject(json)
        val m = obj.getJSONArray("mean")
        val s = obj.getJSONArray("std")
        for (i in 0 until 6) {
            mean[i] = m.getDouble(i).toFloat()
            std[i] = s.getDouble(i).toFloat()
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
