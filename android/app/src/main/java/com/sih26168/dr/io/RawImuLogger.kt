package com.sih26168.dr.io

import android.content.Context
import android.os.Environment
import androidx.documentfile.provider.DocumentFile
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.io.OutputStreamWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Raw 100Hz IMU collection logger — Step 7 data-collection mode (brainstorm #1).
 *
 * Writes EVERY sensor sample (not the 10Hz debug cadence of [CsvLogger]) with
 * event-time timestamps, so sessions can be retrained on later:
 *
 *   ts_nanos, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, g_lat, g_lon, g_speed
 *
 * - `ts_nanos` is sensor event time (monotonic — immune to wall-clock jumps;
 *   the header line carries the utc anchor for conversion).
 * - GPS columns are the LATEST fix (1Hz), blank until the first fix — labels
 *   come from interpolation across them, which requires no gaps in the IMU stream.
 * - Location resolves through [StoragePrefs] (default app dir or picked folder).
 */
class RawImuLogger(context: Context, private val storage: StoragePrefs) {
    companion object {
        const val FLUSH_EVERY_ROWS = 500  // ~5s at 100Hz
    }

    private val appContext = context.applicationContext
    private var writer: BufferedWriter? = null
    private var safDoc: DocumentFile? = null
    private var rowsSinceFlush = 0
    var enabled = false; private set
    /** Rows written this session — for the health sheet. */
    var rowCount = 0L; private set

    /** Start a new session file. Idempotent — stops any previous session first. */
    fun start(deviceModel: String = android.os.Build.MODEL) {
        stop()
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val target = storage.sessionFile(appContext, "imu_log_$ts.csv")
        val utcMs = System.currentTimeMillis()
        writer = BufferedWriter(OutputStreamWriter(target.openOutput(appContext), Charsets.UTF_8), 1 shl 16).apply {
            write("ts_nanos,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,g_lat,g_lon,g_speed\n")
            write("# utc_epoch_ms=$utcMs device=$deviceModel\n")
        }
        // Keep the SAF doc reference for size reporting later.
        safDoc = target.safUri?.let { DocumentFile.fromSingleUri(appContext, it) }
        rowCount = 0
        enabled = true
    }

    /**
     * Log one IMU sample at sensor-event time. GPS fields are the latest fix
     * (or blank before the first fix). One StringBuilder pass per row — safe at 100Hz.
     */
    fun log(
        tsNanos: Long,
        acc: DoubleArray,
        gyro: DoubleArray,
        gLat: Double?, gLon: Double?, gSpeed: Float?,
    ) {
        val w = writer ?: return
        val sb = StringBuilder(96)
        sb.append(tsNanos)
        sb.append(',')
        sb.append(acc[0]); sb.append(',')
        sb.append(acc[1]); sb.append(',')
        sb.append(acc[2]); sb.append(',')
        sb.append(gyro[0]); sb.append(',')
        sb.append(gyro[1]); sb.append(',')
        sb.append(gyro[2]); sb.append(',')
        if (gLat != null) sb.append(gLat)
        sb.append(',')
        if (gLon != null) sb.append(gLon)
        sb.append(',')
        if (gSpeed != null) sb.append(gSpeed)
        sb.append('\n')
        w.write(sb.toString())
        rowCount++
        if (++rowsSinceFlush >= FLUSH_EVERY_ROWS) {
            rowsSinceFlush = 0
            w.flush()
        }
    }

    /** Bytes written this session (best-effort — SAF stat can fail silently). */
    fun currentSizeBytes(): Long = try {
        safDoc?.length() ?: -1L
    } catch (_: Exception) { -1L }

    fun stop() {
        enabled = false
        try { writer?.flush(); writer?.close() } catch (_: Exception) {}
        writer = null
        safDoc = null
    }
}
