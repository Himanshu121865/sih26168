package com.sih26168.dr.engine

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * AVNetLite on-device inference — mirrors python/models/avnet.py AVNetLite.
 * Input: (1,200,6) float32 [acc(3)+gyro(3) normalized]
 * Outputs: v_pred(1,1), log_sig_v(1,1), att_pred(1,3), log_sig_att(1,3), h(1,64)
 * Model file: assets/model.tflite (copied from repo root by gradle task).
 */
class AVNetInference(context: Context) {

    companion object {
        const val WINDOW = 200
        const val CHANNELS = 6
        private const val MODEL_FILE = "model.tflite"
    }

    private val interpreter: Interpreter
    private val inputBuf: ByteBuffer = ByteBuffer
        .allocateDirect(WINDOW * CHANNELS * 4)
        .order(ByteOrder.nativeOrder())

    // 200-sample ring buffer of normalized 6ch @100Hz
    private val ring = Array(WINDOW) { FloatArray(CHANNELS) }
    private var ringHead = 0
    private var filled = 0
    private var samplesSinceInfer = 0
    private val inferEvery = 10   // 100Hz push -> 10Hz inference

    /** Latest outputs */
    var vPred = 0f; private set
    var sigmaV = 1f; private set

    init {
        val model: MappedByteBuffer = context.assets.openFd(MODEL_FILE).use { fd ->
            fd.createInputStream().channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
        }
        val options = Interpreter.Options()
        options.setNumThreads(2)
        interpreter = Interpreter(model, options)
    }

    /** Push one normalized 6ch sample; call at 100Hz. Returns true when a new inference ran (10Hz). */
    fun push(raw: FloatArray): Boolean {
        System.arraycopy(raw, 0, ring[ringHead], 0, CHANNELS)
        ringHead = (ringHead + 1) % WINDOW
        if (filled < WINDOW) filled++
        if (filled < WINDOW) return false
        samplesSinceInfer++
        if (samplesSinceInfer >= inferEvery) {
            samplesSinceInfer = 0
            runInference()
            return true
        }
        return false
    }

    private fun runInference() {
        inputBuf.rewind()
        // fill in time order: oldest first
        for (k in 0 until WINDOW) {
            val idx = (ringHead + k) % WINDOW
            val s = ring[idx]
            for (c in 0 until CHANNELS) inputBuf.putFloat(s[c])
        }
        inputBuf.rewind()
        val vOut = Array(1) { FloatArray(1) }
        val lsOut = Array(1) { FloatArray(1) }
        val attOut = Array(1) { FloatArray(3) }
        val lsAttOut = Array(1) { FloatArray(3) }
        val hOut = Array(1) { FloatArray(64) }
        val outputs = mapOf(0 to vOut, 1 to lsOut, 2 to attOut, 3 to lsAttOut, 4 to hOut)
        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuf), outputs)
        vPred = vOut[0][0]
        sigmaV = kotlin.math.exp(lsOut[0][0])
    }

    fun close() = interpreter.close()
}
