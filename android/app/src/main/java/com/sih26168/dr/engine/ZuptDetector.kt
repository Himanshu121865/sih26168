package com.sih26168.dr.engine

/**
 * Stationary detection + ZUPT gate — port of python/utils/zupt.py
 * (adapted from harshkumarsingh12/dead-reckoning zupt.py).
 * Vehicle thresholds @100Hz; ZUPT: v=0 with tight R when stationary.
 */
class ZuptDetector(
    private val windowSize: Int = 50,          // 0.5s @100Hz
    private val accVarThresh: Double = 0.05,
    private val gyroVarThresh: Double = 0.01,
    private val minDurationS: Double = 0.3,
    private val speedThresh: Double = 0.5,
) {
    private val accNorms = ArrayDeque<Double>(windowSize)
    private val gyroNorms = ArrayDeque<Double>(windowSize)
    private var candidateStartS: Double? = null
    private var nowS = 0.0
    var isStationary = false; private set

    /** aBody/wBody raw samples; speed = AI forward speed (m/s). Call at 100Hz. */
    fun update(aBody: DoubleArray, wBody: DoubleArray, dt: Double, speed: Double?): Boolean {
        nowS += dt
        accNorms.addLast(LieGroup.norm(aBody))
        gyroNorms.addLast(LieGroup.norm(wBody))
        if (accNorms.size > windowSize) accNorms.removeFirst()
        if (gyroNorms.size > windowSize) gyroNorms.removeFirst()
        if (accNorms.size < windowSize) { isStationary = false; return false }
        val aVar = variance(accNorms)
        val wVar = variance(gyroNorms)
        val speedOk = speed == null || speed < speedThresh
        if (aVar < accVarThresh && wVar < gyroVarThresh && speedOk) {
            if (candidateStartS == null) candidateStartS = nowS
            isStationary = (nowS - candidateStartS!!) >= minDurationS
        } else {
            candidateStartS = null
            isStationary = false
        }
        return isStationary
    }

    private fun variance(d: ArrayDeque<Double>): Double {
        var m = 0.0
        for (x in d) m += x
        m /= d.size
        var v = 0.0
        for (x in d) { val e = x - m; v += e * e }
        return v / d.size
    }
}
