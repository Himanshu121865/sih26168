package com.sih26168.dr.engine

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.sin

/**
 * Bike lean detector + adaptive NHC — port of python/models/lean_estimator.py (physics path).
 * p_bike currently fixed 0.0 (car mode) until the bike classifier is trained (Step 8).
 */
class LeanDetector(private val alpha: Double = 0.02) {
    // low-passed gravity estimate in body frame — exposed for DrPipeline linear acc
    val gEst = doubleArrayOf(0.0, 0.0, 9.81)
    var phi = 0.0; private set          // lean angle rad
    var pBike = 0.0; private set

    /** Update with one raw accel sample (m/s^2). */
    fun update(acc: DoubleArray) {
        for (i in 0..2) gEst[i] += alpha * (acc[i] - gEst[i])
        // lean about forward axis (body x): atan2(acc_y, acc_z) per python lean_estimator
        var p = atan2(gEst[1], gEst[2])
        val lim = Math.toRadians(40.0)
        p = p.coerceIn(-lim, lim)
        phi = p
    }

    /**
     * Adaptive NHC (python nhc_correction):
     *   car:  v_lat = 0, R scale 1.0
     *   bike: v_lat = v_fwd * sin(phi), R scale 1 + 2|phi|
     */
    fun nhc(vFwd: Double, isBike: Boolean): Pair<Double, Double> {
        return if (isBike) {
            Pair(vFwd * sin(phi), 1.0 + 2.0 * abs(phi))
        } else {
            Pair(0.0, 1.0)
        }
    }
}
