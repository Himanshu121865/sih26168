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
    val ekf = InEKFEngine(useGravity = false)   // we feed gravity-removed linear acc; engine must NOT add G again
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
    /** Rate-limited + debounced forward speed used for DR. Exposed for UI/position. */
    val smoothedV: Double get() = velSmooth

    companion object {
        const val ZUPT_HOLD_MS = 1300.0             // hold v=0 this long after any gate fires
        const val MOTION_CONFIRM_MS = 500.0        // sustained un-gated motion before trusting model
        const val V_DEADBAND = 0.22               // m/s — below this, don't move position
        const val V_STEP_LIMIT = 0.35          // max |Δv| per 0.1s tick (~3.5 m/s^2)
        const val V_SMOOTH_ALPH = 0.15          // ~0.8s low-pass at 10Hz (1-exp(-0.1/0.8))
    }
    private var velSmooth = 0.0
    private var lastModelV = 0.0
    private var motionConfirmMs = 0.0
    private var zuptHoldMsRem = 0.0

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
            // ~ every 0.1s. ZUPT + table/hand gates — a 2cm table nudge must be 0 m of travel.
            val linAccMag = Math.sqrt(linAcc[0]*linAcc[0] + linAcc[1]*linAcc[1] + linAcc[2]*linAcc[2])
            val gyroMag = Math.sqrt(gyro[0]*gyro[0] + gyro[1]*gyro[1] + gyro[2]*gyro[2])
            val imuStill = zupt.update(acc, gyro, dt, null)
            // Flat on table: any motion is table, not vehicle (low-pass gEst is stable).
            val isFlatOnTable = Math.abs(g[2]) > 7.5
            val isSmallMotion = linAccMag < 1.0 && gyroMag < 0.5
            val isPureRotation = gyroMag > 0.8 && linAccMag < 0.8
            val isVerticalShake = Math.abs(linAcc[2]) > 2.0 && Math.abs(linAcc[0]) < 1.5 && Math.abs(linAcc[1]) < 1.5
            val isLateralTableSlide = isFlatOnTable && Math.abs(linAcc[1]) > 1.0 && Math.abs(linAcc[1]) > Math.abs(linAcc[0])
            val gate = imuStill || isFlatOnTable || isSmallMotion || isPureRotation || isVerticalShake || isLateralTableSlide

            val rawV = max(avnet.vPred.toDouble(), 0.0)

            // 1) Debounce: require sustained un-gated motion before trusting the model
            if (gate) {
                motionConfirmMs = 0.0
            } else if (motionConfirmMs < MOTION_CONFIRM_MS) {
                motionConfirmMs += 100.0
            }
            val trustModel = motionConfirmMs >= MOTION_CONFIRM_MS

            // 2) While gated (or just after), hold v = 0 for ZUPT_HOLD_MS so recovery doesn't spike
            if (gate) zuptHoldMsRem = ZUPT_HOLD_MS
            val vSourced = if (trustModel && zuptHoldMsRem <= 0) rawV else 0.0

            // 3) Rate-limit: a vehicle can't jerk 0 -> 10 m/s inside 100ms
            val step = (vSourced - lastModelV).coerceIn(-V_STEP_LIMIT, V_STEP_LIMIT)
            lastModelV += step

            // 4) Low-pass (a few hundred ms) to kill remaining transients
            velSmooth += V_SMOOTH_ALPH * (lastModelV - velSmooth)

            // 5) Arm motion only if it's a real deadband-crossing hold-0 tail
            if (zuptHoldMsRem > 0) zuptHoldMsRem -= 100

            val v = if (velSmooth > V_DEADBAND) velSmooth else 0.0
            val vLatRaw = lean.nhc(v, lean.pBike > 0.5).first
            val moving = v > 0.0
            val rFwd = if (!moving || gate) 0.05 * 0.05 else max(avnet.sigmaV.toDouble(), 0.3).let { it * it }

            // Propagate with LINEAR acc (gravity already removed) — avoids the 9.8·sin(θ) attitude leak.
            // Skip propagation while gated/stationary to stop noise being integrated into position.
            if (!gate) ekf.propagate(gyro, linAcc, dt)

            val z = doubleArrayOf(v, vLatRaw, 0.0)
            val r = doubleArrayOf(rFwd, rFwd, 25.0)
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
