package com.sih26168.dr.engine

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LieGroupTest {

    @Test
    fun so3exp_zeroIsIdentity() {
        val R = LieGroup.so3exp(doubleArrayOf(0.0, 0.0, 0.0))
        for (i in 0..2) for (j in 0..2) {
            assertEquals(if (i == j) 1.0 else 0.0, R[i][j], 1e-12)
        }
    }

    @Test
    fun so3exp_orthogonal() {
        val R = LieGroup.so3exp(doubleArrayOf(0.1, -0.2, 0.3))
        val Rt = LieGroup.matTranspose(R)
        val I = LieGroup.matMul(R, Rt)
        for (i in 0..2) assertEquals(1.0, I[i][i], 1e-9)
    }

    @Test
    fun sen3exp_matchesPythonSmallAngle() {
        // python: sen3exp([1e-9,0,0, .1,.2,.3, 0,0,0]) -> J≈I + K/2
        val (R, v, p) = LieGroup.sen3exp(
            doubleArrayOf(1e-9, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0)
        )
        assertEquals(0.1, v[0], 1e-6)
        assertEquals(0.2, v[1], 1e-6)
        assertEquals(0.3, v[2], 1e-6)
        for (i in 0..2) for (j in 0..2)
            assertTrue(R[i][j].let { it * it } < 1e-12 || i == j)
    }
}
