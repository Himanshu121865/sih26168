package com.sih26168.dr.io

import android.content.Context
import android.content.SharedPreferences
import android.net.Uri
import android.os.Environment
import androidx.documentfile.provider.DocumentFile
import java.io.File

/**
 * Persisted storage location for collection logs (dev panel setting).
 *
 * Default: the app's external Documents dir (`Android/data/.../Documents`) —
 * always writable, no permission needed, but wiped on uninstall.
 *
 * Custom: a SAF tree URI picked via ACTION_OPEN_DOCUMENT_TREE (e.g. user
 * picks `Documents/SIH26168` or an SD card). Persisted with takePersistableUriPermission
 * so it survives reboots. Appends never need per-file dialogs.
 *
 * Loggers resolve files through [logFile]; SAF path writes via DocumentFile
 * (streaming, no per-row DocumentFile overhead — one file per session).
 */
class StoragePrefs(context: Context) {
    companion object {
        const val PREFS = "storage_prefs"
        const val KEY_TREE_URI = "log_tree_uri"
    }

    private val prefs: SharedPreferences = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private val appContext = context.applicationContext

    /** Custom SAF tree, or null when using the default dir. */
    val treeUri: Uri?
        get() = prefs.getString(KEY_TREE_URI, null)?.let(Uri::parse)

    /** Human-readable location for the dev panel display. */
    fun displayPath(context: Context): String {
        val uri = treeUri ?: return defaultPath(context)
        return uri.lastPathSegment?.replace(':', '/') ?: uri.toString()
    }

    fun defaultPath(context: Context): String {
        val ext = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
        return ext?.absolutePath ?: context.filesDir.absolutePath
    }

    /** True when a custom folder is set. */
    val isCustom: Boolean get() = treeUri != null

    /**
     * Create (or get) a session file in the configured location.
     *
     * Default mode returns a plain [File]; SAF mode returns a document-backed
     * marker whose outputStream flows through [DocumentFile]. Loggers just call
     * [openOutput] — they don't care which mode is active.
     */
    fun sessionFile(context: Context, name: String): SessionTarget {
        val uri = treeUri
        return if (uri == null) {
            val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
                ?: context.filesDir
            SessionTarget(plain = File(dir, name))
        } else {
            val tree = DocumentFile.fromTreeUri(context, uri)
                ?: return SessionTarget(plain = defaultFallbackFile(context, name))
            val doc = tree.createFile("text/csv", name)
                ?: return SessionTarget(plain = defaultFallbackFile(context, name))
            SessionTarget(safUri = doc.uri)
        }
    }

    private fun defaultFallbackFile(context: Context, name: String): File {
        val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir
        return File(dir, name)
    }

    /** Persist a newly picked tree (call after ACTION_OPEN_DOCUMENT_TREE result). */
    fun setTreeUri(context: Context, uri: Uri) {
        context.contentResolver.takePersistableUriPermission(
            uri,
            android.content.Intent.FLAG_GRANT_READ_URI_PERMISSION or
                android.content.Intent.FLAG_GRANT_WRITE_URI_PERMISSION,
        )
        prefs.edit().putString(KEY_TREE_URI, uri.toString()).apply()
    }

    /** Revert to the default app dir. */
    fun clearTreeUri() {
        prefs.edit().remove(KEY_TREE_URI).apply()
    }

    /** List existing session files (imu_, tags_, dr_log_ prefixes) with sizes, newest first. */
    fun listSessionFiles(context: Context): List<Triple<String, Long, Long>> {
        val uri = treeUri
        return if (uri == null) {
            val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS) ?: context.filesDir
            dir.listFiles()
                ?.filter { it.name.startsWith("imu_") || it.name.startsWith("tags_") || it.name.startsWith("dr_log_") }
                ?.sortedByDescending { it.lastModified() }
                ?.map { Triple(it.name, it.length(), it.lastModified()) }
                ?: emptyList()
        } else {
            val tree = DocumentFile.fromTreeUri(context, uri) ?: return emptyList()
            tree.listFiles()
                .filter { it.name != null && (it.name!!.startsWith("imu_") || it.name!!.startsWith("tags_") || it.name!!.startsWith("dr_log_")) }
                .sortedByDescending { it.lastModified() }
                .map { Triple(it.name!!, it.length(), it.lastModified()) }
        }
    }

    data class SessionTarget(val plain: File? = null, val safUri: Uri? = null) {
        /** Open a buffered output stream for this session file. */
        fun openOutput(context: Context): java.io.OutputStream =
            safUri?.let { context.contentResolver.openOutputStream(it) }
                ?: plain!!.outputStream()
    }
}
