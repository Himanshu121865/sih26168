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

    init {
        // P2 refuse-to-run: a scaler/model pair from a mismatched window spec
        // throws here — silent degraded inference is worse than a crash with
        // a clear message. (Scaler's own init also verifies the fingerprint.)
        WindowSpecGuard.verifyScaler(context)
    }

    val scaler = Scaler(context)
    val avnet = AVNetInference(context)
    val lean = LeanDetector()
    val zupt = ZuptDetector()
    val ekf = InEKFEngine(useGravity = false)   // we feed gravity-removed linear acc; engine must NOT add G again
    val alignment = AlignmentEngine()
    val seamless = SeamlessHandler()

    private var roadGraph: RoadGraph? = null
    private var matcher: HmmMapMatcher? = null

    /** Current fusion state (sealed — consume with exhaustive `when`). */
    var mode: SeamlessHandler.FusionMode = SeamlessHandler.FusionMode.GnssAided(trust = 1.0)
        private set
    var lat = 0.0
    var lon = 0.0
    var lastSnappedLat = 0.0
    var lastSnappedLon = 0.0
    var lastV = 0.0; private set
    /** Debug — raw model output (before gates) and last stationary decision. */
    var lastRawModelV = 0.0; private set
    var lastStill = false; private set
    val motionConfirmMsPublic: Int get() = motionConfirmMs.toInt()
    /** Rate-limited + debounced forward speed used for DR. Exposed for UI/position. */
    val smoothedV: Double get() = velSmooth

    companion object {
        // Defaults only — LIVE values live in [tuning] (dev panel edits them without rebuild).
        const val ZUPT_HOLD_MS = 800.0             // hold v=0 this long after a still-gate fires
        const val MOTION_CONFIRM_MS = 250.0        // sustained un-gated motion before trusting model
        const val V_DEADBAND = 0.15                // m/s — below this, don't move position (walking ~0.8+)
        const val V_STEP_LIMIT = 0.5               // max |Δv| per 0.1s tick (~5 m/s^2)
        const val V_SMOOTH_ALPH = 0.25             // ~0.35s low-pass at 10Hz
        // GNSS sanity gate: GPS ground-speed below this (m/s) means we are NOT moving —
        // overrides AI speed. Margin 0.3 covers GPS noise (~0.1-0.2 m/s) + walking start-up.
        const val GNSS_STILL_SPEED = 0.3
        // Latched gate: once GPS says still, AI speed stays blocked until GPS itself says
        // moving (or this max hold expires — guards against a stale gate after GPS dies).
        // Indoor GPS flaps INS<->GNSS every ~1.5s; a short timer hold leaked through every
        // flap (34m fake distance measured). Latch + explicit clear is flap-proof.
        const val GNSS_STILL_MAX_HOLD_MS = 30_000.0
        /** Inference block cadence: 100Hz push / inferEvery=10 = 10Hz. */
        const val INFERENCE_DT = 0.1
    }

    /** Live-tunable gate params — dev panel writes these, engine reads every tick. */
    val tuning = TuningParams()
    private var velSmooth = 0.0
    private var lastModelV = 0.0
    private var motionConfirmMs = 0.0
    private var zuptHoldMsRem = 0.0
    private var gnssStillLatched = false
    private var gnssStillDeadlineMs = 0.0
    /** Engine clock (s), advanced in [onImu] — same base used for gate deadlines. */
    private var engineS = 0.0

    /** Wire the offline road graph when available (bundled asset or extracted). */
    fun setRoadGraph(g: RoadGraph?) {
        roadGraph = g
        matcher = g?.let { HmmMapMatcher(it) }
    }

    /** Called from the GNSS callback: GPS ground speed (m/s) says we are standing still. */
    fun onGnssStill() {
        gnssStillLatched = true
        gnssStillDeadlineMs = engineS * 1000.0 + GNSS_STILL_MAX_HOLD_MS
    }

    /** Called from the GNSS callback: GPS reports real movement — clear the still latch. */
    fun onGnssMoving() {
        gnssStillLatched = false
    }

    /** One raw IMU sample (acc m/s^2 incl. gravity, gyro rad/s) @100Hz. */
    fun onImu(acc: DoubleArray, gyro: DoubleArray, dt: Double) {
        engineS += dt
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
            // ~ every 0.1s (inference rate). HARD stationary gate is variance-based
            // only (0.5s window): table = tiny variance -> still=true -> v=0.
            // Walking/pedestrian motion has real accel+gyro variance -> still=false
            // -> motion flows through. BUGFIX: dt here is the LAST 100Hz sample's dt
            // (~0.01s) but this block runs at 10Hz — feeding it made the detector's
            // internal clock and persistence window 10x too slow (0.3s latch took 3s).
            // The 0.5s variance window (50 samples @10Hz) is preserved by passing dt=0.1.
            val still = zupt.update(acc, gyro, INFERENCE_DT, null)

            val rawV = max(avnet.vPred.toDouble(), 0.0)
            lastRawModelV = rawV
            lastStill = still

            // Motion confirmation: model must report speed above deadband for a few
            // consecutive ticks before trusting it (kills table nudges & spikes).
            if (still || rawV < tuning.vDeadband) {
                motionConfirmMs = 0.0
                if (still) zuptHoldMsRem = tuning.zuptHoldMs
            } else if (motionConfirmMs < tuning.motionConfirmMs) {
                motionConfirmMs += INFERENCE_DT * 1000.0
            }
            // GPS-still latch: blocked until a moving fix clears it. While latched, also
            // reset motion-confirm so un-gating needs a fresh sustained-motion window —
            // otherwise the first tick after un-gate resumes instantly (leak observed).
            val gnssGateActive = tuning.gnssGateEnabled && gnssStillLatched && engineS * 1000.0 < gnssStillDeadlineMs
            if (gnssGateActive) motionConfirmMs = 0.0
            if (!gnssGateActive) gnssStillLatched = false
            val trustModel = motionConfirmMs >= tuning.motionConfirmMs && zuptHoldMsRem <= 0.0 && !gnssGateActive

            val vSourced = if (trustModel) rawV else 0.0

            // Rate-limit: no 0 -> 10 m/s inside 100ms.
            val step = (vSourced - lastModelV).coerceIn(-V_STEP_LIMIT, V_STEP_LIMIT)
            lastModelV += step

            // Low-pass to kill remaining transients.
            velSmooth += V_SMOOTH_ALPH * (lastModelV - velSmooth)

            if (zuptHoldMsRem > 0.0) zuptHoldMsRem -= INFERENCE_DT * 1000.0

            val v = if (velSmooth > tuning.vDeadband) velSmooth else 0.0
            val vLatRaw = lean.nhc(v, lean.pBike > 0.5).first
            val moving = v > 0.0
            val rFwd = if (!moving) 0.05 * 0.05 else max(avnet.sigmaV.toDouble(), 0.3).let { it * it }

            // Propagate with LINEAR acc (gravity already removed) — no 9.8·sin(θ) leak.
            // Skip propagation while stationary so noise isn't integrated into position.
            if (!still) ekf.propagate(gyro, linAcc, dt)

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
