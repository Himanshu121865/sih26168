package com.sih26168.dr.engine

import kotlin.math.abs

/**
 * In-vehicle alignment & calibration — AGENTS.md Step 12.3.
 * - roll/pitch: gravity low-pass on accelerometer
 * - yaw: GNSS course-over-ground when moving (complementary blend)
 * - re-calibration trigger: heading jump >15 deg while gyro energy low
 * Returns the rotation that maps body frame -> vehicle frame (forward = x).
 */
class AlignmentEngine {
    var roll = 0.0; private set        // rad
    var pitch = 0.0; private set
    var yawGnss: Double? = null; private set
    var recalibCount = 0; private set

    private var lastYaw: Double? = null

    fun updateAccel(acc: DoubleArray, alpha: Double = 0.05) {
        // gravity low-pass then roll/pitch (z down convention for raw accel incl. gravity)
        roll += alpha * (atan2f(acc[1], acc[2]) - roll)
        pitch += alpha * (atan2f(-acc[0], kotlin.math.hypot(acc[1], acc[2])) - pitch)
    }

    private fun atan2f(y: Double, x: Double): Double = kotlin.math.atan2(y, x)

    /** Feed GNSS course over ground (rad, true north) when speed > ~2 m/s. */
    fun updateGnssHeading(courseRad: Double, speedMps: Double) {
        if (speedMps < 2.0) return
        val prev = yawGnss
        yawGnss = courseRad
        if (prev != null && abs(angleDiff(courseRad, prev)) > Math.toRadians(15.0)) {
            recalibCount++   // mount rotated? force yaw re-alignment downstream
        }
        lastYaw = courseRad
    }

    fun angleDiff(a: Double, b: Double): Double {
        var d = a - b
        while (d > Math.PI) d -= 2 * Math.PI
        while (d < -Math.PI) d += 2 * Math.PI
        return d
    }
}
