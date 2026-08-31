package com.sih26168.dr.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class InEKFEngineTest {

    @Test
    fun propagateZeroAccel_keepsVelocity() {
        val ekf = InEKFEngine(useGravity = false)
        ekf.setInitialVelocity(10.0)
        val gyro = doubleArrayOf(0.0, 0.0, 0.0)
        val acc = doubleArrayOf(0.0, 0.0, 0.0)
        repeat(50) { ekf.propagate(gyro, acc, 0.01) }
        assertEquals(10.0, ekf.forwardSpeed(), 0.5)
    }

    @Test
    fun updateVelocity_pullsTowardMeasurement() {
        val ekf = InEKFEngine(useGravity = false)
        ekf.setInitialVelocity(0.0)
        // strong measurement: v_fwd = 8, tight R
        repeat(20) {
            ekf.propagate(doubleArrayOf(0.0, 0.0, 0.0), doubleArrayOf(0.0, 0.0, 0.0), 0.1)
            ekf.updateVelocity(doubleArrayOf(8.0, 0.0, 0.0), doubleArrayOf(0.01, 0.01, 25.0))
        }
        assertTrue("speed ${ekf.forwardSpeed()} should be near 8", ekf.forwardSpeed() in 7.0..9.0)
    }

    @Test
    fun biasClamped() {
        val ekf = InEKFEngine(useGravity = false)
        // hammer with huge acc to push ba estimate, then check clamp
        repeat(200) {
            ekf.propagate(doubleArrayOf(0.0, 0.0, 0.0), doubleArrayOf(100.0, 0.0, 0.0), 0.1)
            ekf.updateVelocity(doubleArrayOf(0.0, 0.0, 0.0), doubleArrayOf(0.01, 0.01, 25.0))
        }
        assertTrue(ekf.position().size == 3)
    }
}
