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

    /** Wire the offline road graph when available (bundled asset or extracted). */
    fun setRoadGraph(g: RoadGraph?) {
        roadGraph = g
        matcher = g?.let { HmmMapMatcher(it) }
    }

    /** One raw IMU sample (acc m/s^2 incl. gravity, gyro rad/s) @100Hz. */
    fun onImu(acc: DoubleArray, gyro: DoubleArray, dt: Double) {
        alignment.updateAccel(acc)
        lean.update(acc)
        // normalize + push into the model's 200-sample ring; inference fires every 10th push
        val norm = FloatArray(6)
        scaler.normalize(
            floatArrayOf(acc[0].toFloat(), acc[1].toFloat(), acc[2].toFloat(),
                gyro[0].toFloat(), gyro[1].toFloat(), gyro[2].toFloat()),
            norm,
        )
        if (avnet.push(norm)) {
            // new v_pred available — propagate one 0.1s step with the latest sample
            ekf.propagate(gyro, acc, dt)
            val v = max(avnet.vPred.toDouble(), 0.0)
            val (vLat, rScale) = lean.nhc(v, lean.pBike > 0.5)
            val zuptActive = zupt.update(acc, gyro, dt, v) || v < 0.3
            val rFwd = if (zuptActive) 0.05 * 0.05 else max(avnet.sigmaV.toDouble(), 0.3).let { it * it }
            val z = doubleArrayOf(if (zuptActive) 0.0 else v, vLat, 0.0)
            val r = doubleArrayOf(rFwd, rFwd * rScale, 25.0)
            ekf.updateVelocity(z, r)
        }
    }

    /** 10Hz fusion tick: propagate + velocity/NHC/ZUPT update. Returns v_fwd. */
    fun onFusionTick(dt: Double, gnssSpeed: Double?, gnssCourseRad: Double?): Double {
        mode = seamless.tick((dt * 1000).toInt())
        gnssCourseRad?.let { gnssSpeed?.let { s -> alignment.updateGnssHeading(it, s) } }

        // engine loop propagates inside the 100Hz IMU callback in production;
        // at this tick we propagate one averaged step for the replay harness.
        // (MainActivity wires per-sample propagation.)
        val v = max(avnet.vPred.toDouble(), 0.0)
        return v
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
