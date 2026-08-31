package com.sih26168.dr.engine

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class LeanDetectorTest {

    @Test
    fun bikeLean30deg_vLatIsHalfForward() {
        // mirrors python/inekf_harness.py --test-lean (PASS case)
        val lean = LeanDetector(alpha = 1.0)  // instant low-pass
        // craft acc so atan2(acc_y, acc_z) = 30deg
        val g = 9.81
        val phi = Math.toRadians(30.0)
        lean.update(doubleArrayOf(0.0, g * sin(phi), g * kotlin.math.cos(phi)))
        val (vLat, rScale) = lean.nhc(5.0, isBike = true)
        assertEquals(5.0 * sin(phi), vLat, 1e-9)
        assertEquals(1.0 + 2.0 * phi, rScale, 1e-9)
    }

    @Test
    fun carFallback_zeroLateral() {
        val lean = LeanDetector()
        val (vLat, rScale) = lean.nhc(5.0, isBike = false)
        assertEquals(0.0, vLat, 1e-12)
        assertEquals(1.0, rScale, 1e-12)
    }
}
