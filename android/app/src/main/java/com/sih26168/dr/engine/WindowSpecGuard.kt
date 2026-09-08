package com.sih26168.dr.engine

import android.content.Context
import org.json.JSONObject

/**
 * Startup guard for the window spec (P2) — refuses inference on mismatch.
 *
 * Reads the SAME fingerprint fields Python stamps into `assets/scaler.json`
 * (`spec_version`, `spec_sha256`) and compares against the expected values
 * compiled into this APK. A stale model/scaler pair (e.g. retrained scaler
 * with an old model, or window-path drift) throws at construction time —
 * fail loud, never silently degraded inference.
 *
 * See docs/WINDOW_SPEC.md and python/core/spec.py (single implementation).
 */
object WindowSpecGuard {

    /** Must match python/core/spec.py SPEC_VERSION at build time. */
    const val EXPECTED_SPEC_VERSION = 2

    /**
     * SHA-256 of the canonical spec text — pinned from
     * `python.core.spec.spec_sha256()` (063703d0fd89…, 2026-09-08).
     * A Python-side SPEC_TEXT change must update this literal AND
     * `tools/gen_golden_vectors.py` in the same PR — `WindowGoldenTest`
     * cross-checks the two via the shared golden asset.
     */
    const val EXPECTED_SPEC_SHA256 =
        "063703d0fd897b5280b2c64257dad28fea6743dc03ac05b9ab62f175a6ed42bd"

    /** Thrown when the bundled scaler/model violate the window contract. */
    class SpecMismatch(message: String) : IllegalStateException(message)

    /**
     * Hard-refuse mode (false until the spec-v2 retrain lands — staged
     * rollout per docs/WINDOW_SPEC.md). Flip to true / wire from
     * STRICT_SPEC when the Studio gate goes live; until then mismatches
     * log loudly (Log.w) instead of crashing the field app.
     */
    @Volatile
    var strict: Boolean = false

    /**
     * Verify the bundled scaler against the compiled-in spec.
     *
     * @param context used to open `assets/scaler.json`.
     * @throws SpecMismatch on missing fields, stale version, or hash drift —
     *   only when [strict] is true; otherwise logs loudly and continues.
     */
    fun verifyScaler(context: Context) {
        val json = context.assets.open("scaler.json").bufferedReader().use { it.readText() }
        val obj = JSONObject(json)
        if (!obj.has("spec_version") || !obj.has("spec_sha256")) {
            val msg = "scaler.json has NO spec fingerprint (spec_version/spec_sha256 " +
                "missing). Regenerate with preprocess.py (spec v$EXPECTED_SPEC_VERSION)."
            if (strict) throw SpecMismatch(msg) else android.util.Log.w("WindowSpecGuard", "WARN: $msg")
            return
        }
        val gotVer = obj.getInt("spec_version")
        if (gotVer != EXPECTED_SPEC_VERSION) {
            val msg = "scaler.json spec_version=$gotVer != expected $EXPECTED_SPEC_VERSION — " +
                "retrain or re-export before pairing."
            if (strict) throw SpecMismatch(msg) else android.util.Log.w("WindowSpecGuard", "WARN: $msg")
            return
        }
        val gotHash = obj.getString("spec_sha256")
        if (gotHash != EXPECTED_SPEC_SHA256) {
            val msg = "scaler.json spec_sha256 mismatch (got ${gotHash.take(12)}) — " +
                "window path drifted from docs/WINDOW_SPEC.md. Refusing inference."
            if (strict) throw SpecMismatch(msg) else android.util.Log.w("WindowSpecGuard", "WARN: $msg")
        }
    }
}
