package com.sih26168.dr.engine

import android.content.Context
import org.json.JSONObject

/** Per-channel mean/std — mirrors python/scaler.json (train-only stats). */
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

    /** raw 6ch -> normalized 6ch (output array reused). */
    fun normalize(raw: FloatArray, out: FloatArray) {
        for (i in 0 until 6) out[i] = (raw[i] - mean[i]) / std[i]
    }
}
