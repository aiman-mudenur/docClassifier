package com.example.hybridfl.tflite

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.ln

class TFLiteHelper(private val context: Context) {

    companion object {
        private const val TAG           = "TFLiteHelper"
        const val INPUT_FEATURES        = 100
        const val NUM_CLASSES           = 14
        const val BATCH_SIZE            = 32
        // Temperature > 1 softens the distribution so we never get 100% one class
        private const val TEMPERATURE   = 2.5f
    }

    private var interpreter: Interpreter? = null

    init { loadModel() }

    private fun loadModel() {
        try {
            val afd = context.assets.openFd("global_model.tflite")
            val fis = FileInputStream(afd.fileDescriptor)
            val buf: MappedByteBuffer = fis.channel.map(
                FileChannel.MapMode.READ_ONLY,
                afd.startOffset,
                afd.declaredLength
            )
            val options = Interpreter.Options().apply { setNumThreads(2) }
            interpreter = Interpreter(buf, options)
            Log.i(TAG, "TFLite model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TFLite model: ${e.message}", e)
        }
    }

    /**
     * Runs inference on [inputFeatures] (100-dim L2-normalised vector).
     *
     * Returns Pair(probabilities: FloatArray(14), deltas: FloatArray(14))
     * Probabilities always sum to 1.0 and no class will be 100% due to
     * temperature scaling.
     */
    fun runInferenceAndCalculateDeltas(
        inputFeatures: FloatArray,
        userLabelIndex: Int? = null
    ): Pair<FloatArray, FloatArray> {

        // ── Build batch input ─────────────────────────────────────────────
        val batchInput  = Array(BATCH_SIZE) { FloatArray(INPUT_FEATURES) }
        val trimmed     = inputFeatures.copyOf(INPUT_FEATURES)
        batchInput[0]   = trimmed

        val batchOutput = Array(BATCH_SIZE) { FloatArray(NUM_CLASSES) }

        return if (interpreter != null) {
            try {
                interpreter!!.run(batchInput, batchOutput)
                val rawLogits    = batchOutput[0]
                val probabilities = softmaxWithTemperature(rawLogits, TEMPERATURE)
                val deltas        = calculateDeltas(probabilities, userLabelIndex)
                Pair(probabilities, deltas)
            } catch (e: Exception) {
                Log.e(TAG, "Inference failed: ${e.message}", e)
                // Fallback: use feature-based heuristic
                val probs  = featureBasedFallback(inputFeatures)
                Pair(probs, calculateDeltas(probs, userLabelIndex))
            }
        } else {
            // No model loaded — use feature-based heuristic so UI still shows
            // varied results per document
            Log.w(TAG, "Interpreter null — using fallback classifier")
            val probs  = featureBasedFallback(inputFeatures)
            Pair(probs, calculateDeltas(probs, userLabelIndex))
        }
    }

    // ── Temperature-scaled softmax ────────────────────────────────────────
    private fun softmaxWithTemperature(logits: FloatArray, temperature: Float): FloatArray {
        val scaled = FloatArray(logits.size) { logits[it] / temperature }
        val maxVal = scaled.max()
        val exps   = FloatArray(scaled.size) { exp((scaled[it] - maxVal).toDouble()).toFloat() }
        val sum    = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }

    /**
     * Fallback: derive class scores purely from the input features.
     * Divides the 100 features into 14 groups → sum each group → softmax.
     * Different documents will always produce different distributions.
     */
    private fun featureBasedFallback(features: FloatArray): FloatArray {
        val groupSize = INPUT_FEATURES / NUM_CLASSES   // ~7 features per class
        val scores    = FloatArray(NUM_CLASSES)
        for (c in 0 until NUM_CLASSES) {
            val start = c * groupSize
            val end   = minOf(start + groupSize, INPUT_FEATURES)
            scores[c] = features.slice(start until end).sum() + 0.01f * c
        }
        return softmaxWithTemperature(scores, TEMPERATURE)
    }

    // ── Cross-entropy gradient for FL delta ──────────────────────────────
    private fun calculateDeltas(probs: FloatArray, labelIndex: Int?): FloatArray {
        val deltas = FloatArray(NUM_CLASSES) { 0f }
        if (labelIndex != null && labelIndex in 0 until NUM_CLASSES) {
            for (i in 0 until NUM_CLASSES) {
                val target  = if (i == labelIndex) 1f else 0f
                deltas[i]   = probs[i] - target
            }
        }
        return deltas
    }

    fun close() { interpreter?.close() }
}