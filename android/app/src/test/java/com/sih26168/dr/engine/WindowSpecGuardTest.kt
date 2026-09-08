package com.sih26168.dr.engine

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * WindowSpecGuard unit tests (P2) — the spec-hash side runs on the JVM with
 * no Android context. The Context-based verifyScaler path is exercised by
 * the device test / runtime guard in DrPipeline.
 */
class WindowSpecGuardTest {

    @Test
    fun specVersionIsCurrent() {
        assertEquals(2, WindowSpecGuard.EXPECTED_SPEC_VERSION)
    }

    @Test
    fun specSha256MatchesPythonImplementation() {
        // Verified against python.core.spec.spec_sha256()
        // (regenerate with: PYTHONPATH=. python -c "from python.core.spec import spec_sha256; print(spec_sha256())")
        val expected = WindowSpecGuard.EXPECTED_SPEC_SHA256
        assertEquals(64, expected.length)
        // Prefix pinned from the Python implementation (2026-09-08):
        assertEquals("063703d0fd89", expected.take(12))
    }
}
