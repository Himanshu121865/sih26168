package com.sih26168.dr.io

import android.content.Context
import androidx.documentfile.provider.DocumentFile
import com.sih26168.dr.engine.BuildInfo
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.io.OutputStreamWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * CSV logger: timestamp, p_pred, p_gnss, v_ai, phi, p_bike, mode — AGENTS.md 12.4.
 *
 * Files land in the [StoragePrefs] location (default app Documents dir or a
 * user-picked folder). Logging is off until [start] is called; every method
 * is a no-op otherwise.
 */
class CsvLogger(private val storage: StoragePrefs) {
    private var file: File? = null
    private var writer: BufferedWriter? = null
    var enabled = false

    /**
     * Start a new log file.
     *
     * Row 1 is the stable header (scoring reads it — never reorder). Row 2 is
     * a `#` run comment with build identity for triage; parsers must skip
     * `#`-prefixed lines (see `docs/INTERFACE_CONTRACTS.md` §3).
     *
     * @param specVersion window-spec version (BuildInfo.SPEC_VERSION).
     * @param modelHash short asset hash of model.tflite, or "?" if unknown.
     * @param scalerHash short asset hash of scaler.json, or "?" if unknown.
     */
    fun start(
        context: Context,
        specVersion: Int = BuildInfo.SPEC_VERSION,
        modelHash: String = "?",
        scalerHash: String = "?",
    ) {
        stop()
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val target = storage.sessionFile(context, "dr_log_$ts.csv")
        writer = BufferedWriter(OutputStreamWriter(target.openOutput(context), Charsets.UTF_8)).apply {
            write("timestamp_s,x_pred,y_pred,p_gnss_lat,p_gnss_lon,v_ai,sigma_v,phi_rad,p_bike,mode\n")
            write("# run spec=v$specVersion model=$modelHash scaler=$scalerHash\n")
            flush()
        }
        // Plain-file path kept for the synchronized append fast-path below.
        file = target.plain
        enabled = true
    }

    fun stop() {
        enabled = false
        try { writer?.flush(); writer?.close() } catch (_: Exception) {}
        writer = null
        file = null
    }

    fun log(
        tS: Double, xPred: Double, yPred: Double,
        gLat: Double?, gLon: Double?,
        vAi: Double, sigmaV: Double, phi: Double, pBike: Double, mode: String,
    ) {
        if (!enabled) return
        // Called from BOTH the 10Hz engine ticker and the GNSS callback (~1Hz,
        // different threads) — synchronize on the writer so CSV lines never interleave.
        val line = "$tS,$xPred,$yPred,${gLat ?: ""},${gLon ?: ""},$vAi,$sigmaV,$phi,$pBike,$mode\n"
        val w = writer ?: return
        synchronized(w) { w.write(line) }
    }
}
