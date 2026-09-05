package com.sih26168.dr.engine

/**
 * Seamless GNSS deficit handler — AGENTS.md Step 12.2.
 *
 * Loss: 300ms without fix -> [FusionMode.DeadReckoning], GNSS covariance ramps
 * to infinity. Reacquire: soft reset, blending trust ramps 0->1 over 1s.
 *
 * The fusion state is a sealed interface so consumers handle every case
 * exhaustively with `when` (no `else` needed — the compiler enforces it).
 */
class SeamlessHandler(
    private val lossAfterMs: Long = 300,
    private val rRampMs: Long = 200,
    private val reacquireBlendMs: Long = 1000,
) {
    /**
     * Fusion state.
     *
     * Trust lives inside [GnssAided] (0 = just reacquired, 1 = fully trusted)
     * instead of a parallel field, so state and trust can never disagree.
     */
    sealed interface FusionMode {
        /** Stable short label for logs/debugging ("GNSS" / "INS"). */
        val displayName: String

        /** GNSS-aided navigation.
         *
         * @param trust how much GNSS position to blend in, 0..1.
         */
        data class GnssAided(val trust: Double) : FusionMode {
            override val displayName: String = "GNSS"
        }

        /** Dead reckoning — GNSS lost, AVNet + NHC/ZUPT only. */
        data object DeadReckoning : FusionMode {
            override val displayName: String = "INS"
        }
    }

    /** Current fusion state. */
    var mode: FusionMode = FusionMode.GnssAided(trust = 1.0)
        private set

    private var lastFixElapsedMs: Long = 0
    private var reacquireStartMs: Long = -1
    private var nowMs: Long = 0

    /**
     * Record a GNSS fix.
     *
     * @param nowElapsedMs monotonic clock in ms (same base as [tick] accumulates).
     */
    fun onFix(nowElapsedMs: Long) {
        nowMs = nowElapsedMs
        if (mode is FusionMode.DeadReckoning) {
            reacquireStartMs = nowMs // start soft-blend
            mode = FusionMode.GnssAided(trust = 0.0)
        }
        lastFixElapsedMs = nowElapsedMs
    }

    /**
     * Advance the engine clock; call every engine loop.
     *
     * @param dtMs elapsed ms since the last tick.
     * @return the current [FusionMode].
     */
    fun tick(dtMs: Int): FusionMode {
        nowMs += dtMs
        val current = mode
        if (current is FusionMode.GnssAided) {
            if (nowMs - lastFixElapsedMs > lossAfterMs) {
                mode = FusionMode.DeadReckoning
            } else if (reacquireStartMs >= 0) {
                val t = (nowMs - reacquireStartMs).toDouble() / reacquireBlendMs
                mode = current.copy(trust = t.coerceIn(0.0, 1.0))
                if ((mode as FusionMode.GnssAided).trust >= 1.0) reacquireStartMs = -1
            }
        } else {
            // (not used in INS — reacquire handled in onFix)
        }
        return mode
    }

    /** GNSS position covariance scale for the filter: infinity when INS. */
    fun gnssRScale(): Double = when (val m = mode) {
        is FusionMode.DeadReckoning -> 1e9
        is FusionMode.GnssAided -> 1.0 / kotlin.math.max(m.trust, 1e-3)
    }
}
