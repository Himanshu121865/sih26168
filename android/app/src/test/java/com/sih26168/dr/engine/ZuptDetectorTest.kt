package com.sih26168.dr.engine

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

class ZuptDetectorTest {

    private fun stillSample(rng: Random): Pair<DoubleArray, DoubleArray> =
        doubleArrayOf(rng.nextDouble(-0.02, 0.02), rng.nextDouble(-0.02, 0.02), 0.2) to
            doubleArrayOf(rng.nextDouble(-0.005, 0.005), rng.nextDouble(-0.005, 0.005), rng.nextDouble(-0.005, 0.005))

    @Test
    fun stillStream_becomesStationary() {
        val det = ZuptDetector()
        val rng = Random(0)
        var last = false
        repeat(100) {
            val (a, w) = stillSample(rng)
            last = det.update(a, w, dt = 0.01, speed = 0.0)
        }
        assertTrue(last)
        assertTrue(det.isStationary)
    }

    @Test
    fun movingStream_neverStationary() {
        val det = ZuptDetector()
        val rng = Random(1)
        var ever = false
        repeat(100) {
            val a = doubleArrayOf(rng.nextDouble(-2.0, 2.0), rng.nextDouble(-2.0, 2.0), rng.nextDouble(-2.0, 2.0))
            val w = doubleArrayOf(rng.nextDouble(-0.5, 0.5), rng.nextDouble(-0.5, 0.5), rng.nextDouble(-0.5, 0.5))
            ever = ever or det.update(a, w, dt = 0.01, speed = 5.0)
        }
        assertFalse(ever)
    }

    @Test
    fun speedGate_blocksOtherwiseStillSamples() {
        val det = ZuptDetector()
        val rng = Random(0)
        var ever = false
        repeat(100) {
            val (a, w) = stillSample(rng)
            ever = ever or det.update(a, w, dt = 0.01, speed = 10.0)
        }
        assertFalse(ever)
    }

    @Test
    fun partialWindow_reportsNotStationary() {
        val det = ZuptDetector(windowSize = 50)
        val rng = Random(0)
        repeat(10) {
            val (a, w) = stillSample(rng)
            assertFalse(det.update(a, w, dt = 0.01, speed = 0.0))
        }
    }
}
