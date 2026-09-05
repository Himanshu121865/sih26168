package com.sih26168.dr.engine

import kotlin.math.abs

/**
 * In-vehicle alignment & calibration — AGENTS.md Step 12.3.
 *
 * - roll/pitch from a gravity low-pass on the accelerometer;
 * - yaw from GNSS course-over-ground above ~2 m/s (complementary blend);
 * - re-calibration counter bumps on heading jumps >15° (mount rotated?).
 */
class AlignmentEngine {
    /** Roll in radians (gravity low-pass). */
    var roll = 0.0; private set        // rad
    /** Pitch in radians (gravity low-pass). */
    var pitch = 0.0; private set
    /** Latest GNSS course in radians, or null before the first fast fix. */
    var yawGnss: Double? = null; private set
    /** Count of >15° heading jumps observed (mount-rotation signal). */
    var recalibCount = 0; private set

    private var lastYaw: Double? = null

    /**
     * Feed one raw accelerometer sample (m/s² incl. gravity).
     *
     * @param acc acceleration triple, size 3.
     * @param alpha low-pass coefficient.
     */
    fun updateAccel(acc: DoubleArray, alpha: Double = 0.05) {
        // gravity low-pass then roll/pitch (z down convention for raw accel incl. gravity)
        roll += alpha * (atan2f(acc[1], acc[2]) - roll)
        pitch += alpha * (atan2f(-acc[0], kotlin.math.hypot(acc[1], acc[2])) - pitch)
    }

    private fun atan2f(y: Double, x: Double): Double = kotlin.math.atan2(y, x)

    /**
     * Feed GNSS course-over-ground in radians (true north).
     *
     * Ignored below ~2 m/s where course is noise.
     *
     * @param courseRad course in radians.
     * @param speedMps vehicle speed in m/s.
     */
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
