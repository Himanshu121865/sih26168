package com.sih26168.dr.engine

import android.content.Context
import com.sih26168.dr.map.HmmMapMatcher
import com.sih26168.dr.map.RoadGraph
import kotlin.math.max

/**
 * Glues the full engine: AVNet inference -> InEKF (NHC/ZUPT) -> map matcher.
 * Called from MainActivity's 100Hz sensor loop (denormalize -> push) and
 * 10Hz fusion tick (predict -> update -> emit pose).
 */
class DrPipeline(context: Context, useGravity: Boolean = true) {

    val scaler = Scaler(context)
    val avnet = AVNetInference(context)
    val lean = LeanDetector()
    val zupt = ZuptDetector()
    val ekf = InEKFEngine(useGravity = useGravity)
    val alignment = AlignmentEngine()
    val seamless = SeamlessHandler()

    private var roadGraph: RoadGraph? = null
    private var matcher: HmmMapMatcher? = null

    var mode: SeamlessHandler.Mode = SeamlessHandler.Mode.GNSS; private set
    var lat = 0.0
    var lon = 0.0
    var lastSnappedLat = 0.0
    var lastSnappedLon = 0.0
    var lastV = 0.0; private set

    /** Wire the offline road graph when available (bundled asset or extracted). */
    fun setRoadGraph(g: RoadGraph?) {
        roadGraph = g
        matcher = g?.let { HmmMapMatcher(it) }
    }

    /** One raw IMU sample (acc m/s^2 incl. gravity, gyro rad/s) @100Hz. */
    fun onImu(acc: DoubleArray, gyro: DoubleArray, dt: Double) {
        alignment.updateAccel(acc)
        lean.update(acc)
        // CRITICAL: model was trained on linear acc (acc - gravity), not raw.
        // Use LeanDetector's low-pass gravity estimate (gEst) to match python/preprocess.py:186.
        val g = lean.gEst
        val linAcc = doubleArrayOf(acc[0] - g[0], acc[1] - g[1], acc[2] - g[2])
        val norm = FloatArray(6)
        scaler.normalize(
            floatArrayOf(linAcc[0].toFloat(), linAcc[1].toFloat(), linAcc[2].toFloat(),
                gyro[0].toFloat(), gyro[1].toFloat(), gyro[2].toFloat()),
            norm,
        )
        if (avnet.push(norm)) {
            // new v_pred available — propagate one 0.1s step with the latest sample
            ekf.propagate(gyro, acc, dt)
            val rawV = max(avnet.vPred.toDouble(), 0.0)
            val (vLatRaw, rScale) = lean.nhc(rawV, lean.pBike > 0.5)
            // ZUPT: IMU variance + small-motion deadband (hand-held table -> 0.0, small shake -> clamp)
            val linAccMag = Math.sqrt(linAcc[0]*linAcc[0] + linAcc[1]*linAcc[1] + linAcc[2]*linAcc[2])
            val gyroMag = Math.sqrt(gyro[0]*gyro[0] + gyro[1]*gyro[1] + gyro[2]*gyro[2])
            val imuStill = zupt.update(acc, gyro, dt, null)
            val isSmallMotion = linAccMag < 1.0 && gyroMag < 0.5
            val zuptActive = imuStill || rawV < 0.3 || isSmallMotion
            // also clamp rawV that is implausibly high for small motion
            val vClamped = if (isSmallMotion && rawV > 2.0) 0.0 else rawV
            val v = if (zuptActive) 0.0 else vClamped
            val vLat = if (zuptActive) 0.0 else vLatRaw
            val rFwd = if (zuptActive) 0.05 * 0.05 else max(avnet.sigmaV.toDouble(), 0.3).let { it * it }
            val z = doubleArrayOf(v, vLat, 0.0)
            val r = doubleArrayOf(rFwd, rFwd * rScale, 25.0)
            ekf.updateVelocity(z, r)
            lastV = v
        }
    }

    /** 10Hz fusion tick: propagate + velocity/NHC/ZUPT update. Returns v_fwd (ZUPT-corrected). */
    fun onFusionTick(dt: Double, gnssSpeed: Double?, gnssCourseRad: Double?): Double {
        mode = seamless.tick((dt * 1000).toInt())
        gnssCourseRad?.let { gnssSpeed?.let { s -> alignment.updateGnssHeading(it, s) } }
        // Return last ZUPT-corrected v from onImu (0.0 when still on table)
        return lastV
    }

    /** After ekf.updateVelocity, call to emit pose + map snap. */
    fun emitPose(gnssLat: Double?, gnssLon: Double?): Pair<Double, Double> {
        // p (6:9) holds nav position in meters — integrate lat/lon incrementally upstream.
        matcher?.let { m ->
            val fix = m.update(lat, lon, null)
            if (fix != null) { lastSnappedLat = fix.lat; lastSnappedLon = fix.lon }
        }
        return Pair(lat, lon)
    }
}
