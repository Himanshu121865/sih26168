package com.sih26168.dr.engine

import com.sih26168.dr.engine.SeamlessHandler.FusionMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SeamlessHandlerTest {

    @Test
    fun startsGnssAidedFullTrust() {
        val h = SeamlessHandler()
        val mode = h.mode
        assertTrue(mode is FusionMode.GnssAided)
        assertEquals(1.0, (mode as FusionMode.GnssAided).trust, 1e-12)
        assertEquals(1.0, h.gnssRScale(), 1e-9)
    }

    @Test
    fun lossAfter300ms_switchesToDeadReckoning() {
        val h = SeamlessHandler()
        repeat(3) { h.tick(100) } // 300ms elapsed — boundary is strict `>`
        assertTrue(h.mode is FusionMode.GnssAided)
        h.tick(100) // 400ms > 300ms
        assertTrue(h.mode is FusionMode.DeadReckoning)
        assertEquals(1e9, h.gnssRScale(), 0.0)
    }

    @Test
    fun reacquire_blendsTrustOver1s() {
        val h = SeamlessHandler()
        repeat(4) { h.tick(100) } // t=400 -> INS
        // Fixes must keep arriving inside the 300ms loss window (production: ~1Hz
        // location updates), otherwise the ramp correctly aborts back to INS.
        var clock = 400L
        h.onFix(clock)
        val trusts = (1..4).map {
            clock += 250
            h.tick(250)
            h.onFix(clock)
            (h.mode as FusionMode.GnssAided).trust
        }
        assertEquals(listOf(0.25, 0.5, 0.75, 1.0), trusts)
    }

    @Test
    fun steadyFixes_neverLeaveGnss() {
        val h = SeamlessHandler()
        var clock = 0L
        repeat(20) {
            clock += 100
            h.onFix(clock)
            assertTrue(h.tick(100) is FusionMode.GnssAided)
        }
    }

    @Test
    fun fusionMode_isExhaustiveWithoutElse() {
        // Compiler enforces all branches — adding a state breaks this test at compile time.
        fun label(mode: FusionMode): String = when (mode) {
            is FusionMode.GnssAided -> "GNSS:${mode.trust}"
            is FusionMode.DeadReckoning -> "INS"
        }
        assertEquals("GNSS:1.0", label(FusionMode.GnssAided(1.0)))
        assertEquals("INS", label(FusionMode.DeadReckoning))
    }
}
