package com.sih26168.dr.io

import android.content.Context
import androidx.documentfile.provider.DocumentFile
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Session tagging for data collection (Step 7) — writes a `tags_<ts>.json`
 * sidecar next to the imu/dr logs (in the [StoragePrefs] location).
 *
 * Two tiers (brainstorm #2):
 *  - MANUAL: developer taps a button (hard_brake / pothole / tunnel / wet / mark)
 *    -> tagged with a ±3s window so short events aren't missed.
 *  - AUTO: thresholds computed in-app (hard_brake from accel, pothole spike)
 *    appended by the pipeline — same schema, source field distinguishes them.
 *
 * Tags reference the RAW LOG's ts_nanos base so they align with imu_log rows
 * after the utc anchor conversion.
 */
class TagManager(private val storage: StoragePrefs) {
    companion object {
        /** Manual tags cover this many ms before the tap (event often noticed late). */
        const val PRE_WINDOW_MS = 3000L
        /** ...and this many after (ongoing event). */
        const val POST_WINDOW_MS = 1000L
        val MANUAL_TAGS = listOf("hard_brake", "pothole", "tunnel", "wet", "lean_turn", "mark")
    }

    private var file: File? = null
    private var safUri: android.net.Uri? = null
    private var sessionStartUtcMs = 0L
    private val tags = JSONArray()

    /** Start a sidecar for a session; ts matches RawImuLogger's file naming. */
    fun start(context: Context) {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        sessionStartUtcMs = System.currentTimeMillis()
        val target = storage.sessionFile(context, "tags_$ts.json")
        file = target.plain
        safUri = target.safUri
        tags.let { while (it.length() > 0) it.remove(0) }
        flush(context)
    }

    /** Add a manual tag effective [PRE_WINDOW_MS] before the tap. */
    fun tag(name: String, elapsedRealtimeNanos: Long, note: String = "") {
        val o = JSONObject()
        o.put("tag", name)
        o.put("source", "manual")
        o.put("t_end_nanos", elapsedRealtimeNanos)
        o.put("t_start_nanos", elapsedRealtimeNanos - (PRE_WINDOW_MS + POST_WINDOW_MS) * 1_000_000)
        o.put("utc_ms", System.currentTimeMillis())
        if (note.isNotEmpty()) o.put("note", note)
        tags.put(o)
        file?.parentFile?.let { flushFrom(it) } ?: run { pendingFlush = true }
    }

    /** Add an auto-detected tag (pipeline thresholds). */
    fun auto(name: String, tStartNanos: Long, tEndNanos: Long, value: Double) {
        val o = JSONObject()
        o.put("tag", name)
        o.put("source", "auto")
        o.put("t_start_nanos", tStartNanos)
        o.put("t_end_nanos", tEndNanos)
        o.put("value", value)
        tags.put(o)
        // auto tags can be frequent — flush lazily to keep rewrite cost low
        if (tags.length() % 10 == 0) pendingFlush = true
    }

    @Volatile private var pendingFlush = false

    fun stop(context: Context) {
        pendingFlush = true
        flush(context)
    }

    /** Context-less flush for plain-file mode (fast path from tag()). */
    private fun flushFrom(dir: File) {
        val f = file ?: return
        val root = JSONObject()
        root.put("session_start_utc_ms", sessionStartUtcMs)
        root.put("count", tags.length())
        root.put("tags", tags)
        f.writeText(root.toString(2))
    }

    /** Full flush — works in both plain and SAF modes. */
    fun flush(context: Context) {
        val root = JSONObject()
        root.put("session_start_utc_ms", sessionStartUtcMs)
        root.put("count", tags.length())
        root.put("tags", tags)
        val json = root.toString(2)
        val f = file
        if (f != null) {
            f.writeText(json)
        } else {
            val uri = safUri ?: return
            context.contentResolver.openOutputStream(uri, "wt")?.use { os ->
                os.write(json.toByteArray())
            }
        }
        pendingFlush = false
    }
}
