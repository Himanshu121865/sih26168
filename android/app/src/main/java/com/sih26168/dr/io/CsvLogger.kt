package com.sih26168.dr.io

import android.content.Context
import android.os.Environment
import com.sih26168.dr.engine.BuildInfo
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * CSV logger: timestamp, p_pred, p_gnss, v_ai, phi, p_bike, mode — AGENTS.md 12.4.
 *
 * Files land in the app's Documents dir as `dr_log_<ts>.csv`. Logging is
 * off until [start] is called; every method is a no-op otherwise.
 */
class CsvLogger(context: Context) {
    private val dir: File = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
        ?: context.filesDir
    private var file: File? = null
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
        specVersion: Int = BuildInfo.SPEC_VERSION,
        modelHash: String = "?",
        scalerHash: String = "?",
    ) {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        file = File(dir, "dr_log_$ts.csv").apply {
            writeText("timestamp_s,x_pred,y_pred,p_gnss_lat,p_gnss_lon,v_ai,sigma_v,phi_rad,p_bike,mode\n")
            appendText("# run spec=v$specVersion model=$modelHash scaler=$scalerHash\n")
        }
        enabled = true
    }

    fun stop() { enabled = false; file = null }

    fun log(
        tS: Double, xPred: Double, yPred: Double,
        gLat: Double?, gLon: Double?,
        vAi: Double, sigmaV: Double, phi: Double, pBike: Double, mode: String,
    ) {
        if (!enabled) return
        val f = file ?: return
        f.appendText(
            "$tS,$xPred,$yPred,${gLat ?: ""},${gLon ?: ""},$vAi,$sigmaV,$phi,$pBike,$mode\n"
        )
    }
}
