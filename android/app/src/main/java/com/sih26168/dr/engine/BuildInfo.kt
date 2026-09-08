package com.sih26168.dr.engine

import android.content.Context
import java.security.MessageDigest

/**
 * Build/asset identity for the on-device health sheet (Tier 3 production item).
 *
 * A field-test failure must be diagnosable from a screenshot: spec version +
 * short hashes of the exact model/scaler assets in use. Compare against
 * `docs/WINDOW_SPEC.md` (P2) and `reports/tflite_diff.txt` when triaging.
 */
object BuildInfo {
    /** Window-spec version this APK was built against (mirrors P2 `spec_version`). */
    const val SPEC_VERSION = WindowSpecGuard.EXPECTED_SPEC_VERSION

    /**
     * Short SHA-256 of a bundled asset for display/logging.
     *
     * @param context used to open assets.
     * @param name asset file name (e.g. `"model.tflite"`).
     * @return first 12 hex chars, or `"?"` when the asset is missing/unreadable.
     */
    fun assetHash12(context: Context, name: String): String {
        return try {
            val digest = MessageDigest.getInstance("SHA-256")
            context.assets.open(name).use { input ->
                val buf = ByteArray(64 * 1024)
                while (true) {
                    val n = input.read(buf)
                    if (n <= 0) break
                    digest.update(buf, 0, n)
                }
            }
            digest.digest().joinToString("") { "%02x".format(it) }.take(12)
        } catch (_: Exception) {
            "?"
        }
    }
}
