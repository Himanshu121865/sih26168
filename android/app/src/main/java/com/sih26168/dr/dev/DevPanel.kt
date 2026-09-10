package com.sih26168.dr.dev

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.core.widget.NestedScrollView
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.sih26168.dr.MainActivity
import com.sih26168.dr.R
import com.sih26168.dr.engine.TuningParams

/**
 * Developer panel — hidden bottom sheet opened by 3 quick taps on the status chip.
 *
 * Four sections, all live (no rebuild):
 *  1. COLLECTION — session start/stop for imu_log + tags, file stats
 *  2. TAGGING — tap-to-mark buttons writing ±3s windows to tags sidecar
 *  3. GATE TUNING — sliders for ZUPT/GNSS-still/deadband + toggles
 *  4. LIVE STATS — Hz measured, rows, GPS fixes, mode, speed, sigma, files
 *
 * Built programmatically (no XML) so it can't drift from the layout file;
 * everything is themed from Material attributes where possible.
 */
@SuppressLint("SetTextI18n")
class DevPanel(private val activity: MainActivity) {

    private var dialog: BottomSheetDialog? = null
    private val tune = activity.pipeline.tuning
    private val sliders = mutableMapOf<String, Pair<SeekBar, TextView>>()

    private var statsView: TextView? = null
    private var storagePathText: TextView? = null

    /** Called by MainActivity every ~1s while panel is open. */
    fun refreshStats() {
        if (dialog?.isShowing != true) return
        statsView?.text = activity.collectDevStats()
    }

    fun show() {
        if (dialog?.isShowing == true) return
        val ctx = activity
        // NestedScrollView (not ScrollView) — cooperates with BottomSheetBehavior's
        // nested-scroll protocol. A plain ScrollView let the sheet's drag gesture
        // steal upward swipes at the top edge and dismiss the panel mid-read.
        val root = NestedScrollView(ctx).apply { overScrollMode = View.OVER_SCROLL_NEVER }
        val col = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(16), dp(20), dp(28))
        }
        root.addView(col)

        col.addView(header("DEV PANEL"))
        col.addView(subtitle("Data collection · tagging · gate tuning — all live"))

        // ---- 1. COLLECTION ----
        col.addView(section("COLLECTION"))
        // Storage location: shows current dir, buttons to pick custom or reset.
        storagePathText = TextView(activity).apply {
            // (assigned to the nullable field for later refresh; local val is this)
            textSize = 11f
            setTextColor(0xFF555555.toInt())
            typeface = android.graphics.Typeface.MONOSPACE
            setPadding(0, dp(2), 0, dp(6))
            text = "→ ${activity.storage.displayPath(activity)}"
        }
        col.addView(storagePathText)
        val folderRow = LinearLayout(ctx).apply { orientation = LinearLayout.HORIZONTAL }
        folderRow.addView(Button(ctx).apply {
            text = "Choose folder…"; textSize = 11f; isAllCaps = false
            setOnClickListener { activity.pickFolder() }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = dp(6) })
        folderRow.addView(Button(ctx).apply {
            text = "Default"; textSize = 11f; isAllCaps = false
            setOnClickListener {
                activity.resetFolder()
                storagePathText?.text = "→ ${activity.storage.displayPath(activity)}"
            }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        col.addView(folderRow)
        col.addRowButton("New collection session") { activity.startCollectionSession() }
        col.addRowButton("Stop session (flush all)") { activity.stopCollectionSession() }
        col.addRowButton("Pull stats to clipboard") { activity.copyStatsToClipboard() }

        // ---- 2. TAGGING ----
        col.addView(section("TAGGING (±3s window)"))
        val tagRow1 = LinearLayout(ctx).apply { orientation = LinearLayout.HORIZONTAL }
        val tagRow2 = LinearLayout(ctx).apply { orientation = LinearLayout.HORIZONTAL }
        for ((i, name) in listOf("hard_brake", "pothole", "tunnel").withIndex()) {
            tagRow1.addView(tagButton(name) { activity.tagNow(name) }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = dp(6) })
        }
        for ((i, name) in listOf("wet", "lean_turn", "mark").withIndex()) {
            tagRow2.addView(tagButton(name) { activity.tagNow(name) }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginEnd = dp(6) })
        }
        col.addView(tagRow1)
        col.addView(tagRow2, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { topMargin = dp(6) })
        col.addView(toggleRow("Auto-tag (hard-brake/pothole)", tune.autoTagEnabled) { tune.autoTagEnabled = it })

        // ---- 3. GATE TUNING ----
        col.addView(section("GATE TUNING (live)"))
        addSlider(col, "v_deadband", "m/s deadband", 0.05, 0.5, tune.vDeadband) { tune.vDeadband = it }
        addSlider(col, "motion_confirm", "confirm ms", 0.0, 1000.0, tune.motionConfirmMs) { tune.motionConfirmMs = it }
        addSlider(col, "zupt_hold", "hold ms", 0.0, 2000.0, tune.zuptHoldMs) { tune.zuptHoldMs = it }
        addSlider(col, "gnss_still", "GPS still m/s", 0.1, 1.0, tune.gnssStillSpeed) { tune.gnssStillSpeed = it }
        addSlider(col, "gnss_moving", "GPS moving m/s", 0.3, 2.0, tune.gnssMovingSpeed) { tune.gnssMovingSpeed = it }
        addSlider(col, "acc_var", "acc var thr", 0.01, 0.5, tune.accVarThresh) { tune.accVarThresh = it }
        addSlider(col, "gyro_var", "gyro var thr", 0.001, 0.1, tune.gyroVarThresh) { tune.gyroVarThresh = it }
        col.addView(toggleRow("GNSS still gate (latch)", tune.gnssGateEnabled) { tune.gnssGateEnabled = it })
        col.addView(section("AUTO-TAG THRESHOLDS"))
        addSlider(col, "hard_brake_g", "hard brake m/s²", 2.0, 8.0, tune.hardBrakeThresh) { tune.hardBrakeThresh = it }
        addSlider(col, "pothole_g", "pothole spike m/s²", 10.0, 60.0, tune.potholeSpikeThresh) { tune.potholeSpikeThresh = it }

        // ---- 4. LIVE STATS ----
        col.addView(section("LIVE STATS"))
        // Fresh TextView per show(): reusing the field crashed on second open
        // ("child already has a parent" — old sheet's hierarchy still held it).
        val statsText = TextView(activity).apply {
            setTextColor(0xFF202020.toInt())
            textSize = 12f
            typeface = android.graphics.Typeface.MONOSPACE
            setTextIsSelectable(true)
            text = activity.collectDevStats()
        }
        statsView = statsText
        col.addView(statsText)

        col.addView(section(""))
        col.addView(Button(ctx).apply {
            text = "Close"
            setOnClickListener { dismiss() }
        })

        dialog = BottomSheetDialog(ctx).apply {
            setContentView(root)
            // No peek/collapsed half-state: either fully open or dismissed. Prevents
            // the "scrolled to top, sheet shrank to a sliver" mid-read state.
            behavior.skipCollapsed = true
            behavior.isFitToContents = true
            window?.findViewById<View>(com.google.android.material.R.id.design_bottom_sheet)
                ?.background = GradientDrawable().apply { setColor(0xFFFFFFFF.toInt()); cornerRadius = dp(24).toFloat() }
            setOnDismissListener { dialog = null }
            show()
        }
    }

    fun dismiss() {
        dialog?.dismiss()
        dialog = null
    }

    // ---- helpers ----

    private fun header(s: String) = TextView(activity).apply {
        text = s; textSize = 18f; setTypeface(typeface, android.graphics.Typeface.BOLD)
        setTextColor(0xFF1A1A1A.toInt())
    }

    private fun subtitle(s: String) = TextView(activity).apply {
        text = s; textSize = 12f; setTextColor(0xFF666666.toInt())
    }

    private fun section(s: String) = TextView(activity).apply {
        if (s.isNotEmpty()) text = s
        textSize = 13f; setTypeface(typeface, android.graphics.Typeface.BOLD)
        setTextColor(0xFF444444.toInt())
        setPadding(0, dp(18), 0, dp(6))
    }

    private fun LinearLayout.addRowButton(label: String, onClick: () -> Unit) {
        addView(Button(activity).apply { text = label; setOnClickListener { onClick(); Toast.makeText(activity, label, Toast.LENGTH_SHORT).show() } })
    }

    private fun tagButton(name: String, onClick: () -> Unit) = Button(activity).apply {
        text = name
        textSize = 11f
        isAllCaps = false
        setOnClickListener { onClick(); Toast.makeText(activity, "tagged: $name", Toast.LENGTH_SHORT).show() }
    }

    private fun toggleRow(label: String, initial: Boolean, onChange: (Boolean) -> Unit): View {
        val row = LinearLayout(activity).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        row.addView(CheckBox(activity).apply {
            text = label; isChecked = initial
            setOnCheckedChangeListener { _, checked -> onChange(checked) }
        })
        return row
    }

    @SuppressLint("SetTextI18n")
    private fun addSlider(
        col: LinearLayout, key: String, label: String,
        min: Double, max: Double, initial: Double, onChange: (Double) -> Unit,
    ) {
        val valueText = TextView(activity).apply {
            textSize = 11f; setTextColor(0xFF333333.toInt())
            text = "$label: ${"%.3f".format(initial)}"
        }
        val bar = SeekBar(activity).apply {
            this.max = 1000
            this.progress = (((initial - min) / (max - min)) * 1000).toInt().coerceIn(0, 1000)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                    val v = min + (max - min) * progress / 1000.0
                    valueText.text = "$label: ${"%.3f".format(v)}"
                    onChange(v)
                }
                override fun onStartTrackingTouch(sb: SeekBar?) {}
                override fun onStopTrackingTouch(sb: SeekBar?) {}
            })
        }
        col.addView(valueText)
        col.addView(bar)
        sliders[key] = bar to valueText
    }

    private fun dp(v: Int): Int = (v * activity.resources.displayMetrics.density).toInt()
}
