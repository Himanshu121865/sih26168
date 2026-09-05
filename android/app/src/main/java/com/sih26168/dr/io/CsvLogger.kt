package com.sih26168.dr.io

import android.content.Context
import android.os.Environment
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

    fun start() {
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        file = File(dir, "dr_log_$ts.csv").apply {
            writeText("timestamp_s,x_pred,y_pred,p_gnss_lat,p_gnss_lon,v_ai,sigma_v,phi_rad,p_bike,mode\n")
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
