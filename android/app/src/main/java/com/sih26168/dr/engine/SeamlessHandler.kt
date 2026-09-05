package com.sih26168.dr.engine

import kotlin.math.max

enum class DrMode { GNSS, INS }

/**
 * Seamless GNSS deficit handler — AGENTS.md Step 12.2.
 * Loss: 300ms without fix -> INS mode, R_gnss ramps to infinity over 200ms.
 * Reacquire: soft reset, blending weight alpha ramps 0->1 over 1s.
 */
class SeamlessHandler(
    private val lossAfterMs: Long = 300,
    private val rRampMs: Long = 200,
    private val reacquireBlendMs: Long = 1000,
) {
    var mode = DrMode.GNSS; private set
    /** 0..1 — how much GNSS position to trust right now. */
    var gnssTrust = 1.0; private set

    private var lastFixElapsedMs: Long = 0
    private var reacquireStartMs: Long = -1
    private var nowMs: Long = 0

    fun onFix(nowElapsedMs: Long) {
        nowMs = nowElapsedMs
        if (mode == DrMode.INS) {
            reacquireStartMs = nowMs   // start soft-blend
            mode = DrMode.GNSS
        }
        lastFixElapsedMs = nowElapsedMs
    }

    /** tick(dtMs) every engine loop; returns current mode. */
    fun tick(dtMs: Int): DrMode {
        nowMs += dtMs
        if (mode == DrMode.GNSS) {
            val since = nowMs - lastFixElapsedMs
            if (since > lossAfterMs) {
                mode = DrMode.INS
                gnssTrust = 0.0
            }
        } else {
            // (not used in INS — reacquire handled in onFix)
        }
        if (mode == DrMode.GNSS && reacquireStartMs >= 0) {
            val t = (nowMs - reacquireStartMs).toDouble() / reacquireBlendMs
            gnssTrust = t.coerceIn(0.0, 1.0)
            if (gnssTrust >= 1.0) reacquireStartMs = -1
        }
        return mode
    }

    /** GNSS position covariance scale for the filter: infinity when INS. */
    fun gnssRScale(): Double = if (mode == DrMode.INS) 1e9 else 1.0 / max(gnssTrust, 1e-3)
}
