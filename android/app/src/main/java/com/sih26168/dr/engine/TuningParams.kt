package com.sih26168.dr.engine

/**
 * Live-tunable gate parameters for field calibration (dev panel).
 *
 * Everything here is what the dev panel sliders edit — a session's best
 * settings can be exported to a `tuning_<ts>.json` sidecar and later baked
 * into DrPipeline defaults once validated. Grouped in one mutable holder so
 * the panel reads/writes a single object and the pipeline pulls from it
 * every tick (no rebuild needed).
 */
class TuningParams {
    // --- Speed gates (DrPipeline) ---
    @Volatile var vDeadband: Double = DrPipeline.V_DEADBAND
    @Volatile var motionConfirmMs: Double = DrPipeline.MOTION_CONFIRM_MS
    @Volatile var zuptHoldMs: Double = DrPipeline.ZUPT_HOLD_MS
    @Volatile var gnssStillSpeed: Double = DrPipeline.GNSS_STILL_SPEED
    @Volatile var gnssMovingSpeed: Double = 0.5       // latch clear hysteresis (MainActivity)
    @Volatile var gnssGateEnabled: Boolean = true

    // --- ZUPT detector (ZuptDetector) ---
    @Volatile var accVarThresh: Double = 0.05
    @Volatile var gyroVarThresh: Double = 0.01
    @Volatile var zuptMinDurationS: Double = 0.3

    // --- Collection ---
    @Volatile var autoTagEnabled: Boolean = true
    @Volatile var hardBrakeThresh: Double = 4.4       // m/s^2 (~0.45g) per PS
    @Volatile var potholeSpikeThresh: Double = 25.0   // m/s^2 (~2.5g)

    fun snapshot(): Map<String, String> = mapOf(
        "v_deadband" to vDeadband.toString(),
        "motion_confirm_ms" to motionConfirmMs.toString(),
        "zupt_hold_ms" to zuptHoldMs.toString(),
        "gnss_still_speed" to gnssStillSpeed.toString(),
        "gnss_moving_speed" to gnssMovingSpeed.toString(),
        "gnss_gate_enabled" to gnssGateEnabled.toString(),
        "acc_var_thresh" to accVarThresh.toString(),
        "gyro_var_thresh" to gyroVarThresh.toString(),
        "zupt_min_duration_s" to zuptMinDurationS.toString(),
        "auto_tag_enabled" to autoTagEnabled.toString(),
        "hard_brake_thresh" to hardBrakeThresh.toString(),
        "pothole_spike_thresh" to potholeSpikeThresh.toString(),
    )
}
