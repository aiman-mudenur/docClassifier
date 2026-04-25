package com.example.hybridfl.tflite

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteHelper(private val context: Context) {

    private var interpreter: Interpreter? = null

    companion object {
        const val INPUT_FEATURES = 100   // matches model input shape [32, 100]
        const val NUM_CLASSES = 14       // AG News 14 classes
        const val BATCH_SIZE = 32        // model expects batch of 32
    }

    init { loadModel() }

    private fun loadModel() {
        try {
            val afd = context.assets.openFd("global_model.tflite")
            val fis = FileInputStream(afd.fileDescriptor)
            val channel = fis.channel
            val buf: MappedByteBuffer = channel.map(
                FileChannel.MapMode.READ_ONLY,
                afd.startOffset,
                afd.declaredLength
            )
            val options = Interpreter.Options().apply { setNumThreads(2) }
            interpreter = Interpreter(buf, options)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun runInferenceAndCalculateDeltas(
        inputFeatures: FloatArray,
        userProvidedLabelIndex: Int?
    ): Pair<FloatArray, FloatArray> {

        // Model needs batch of 32 — pad single sample with zeros
        val batchInput = Array(BATCH_SIZE) { FloatArray(INPUT_FEATURES) }
        val trimmed = inputFeatures.take(INPUT_FEATURES).toFloatArray()
        batchInput[0] = if (trimmed.size == INPUT_FEATURES) trimmed
        else FloatArray(INPUT_FEATURES).also {
            trimmed.copyInto(it)
        }

        val batchOutput = Array(BATCH_SIZE) { FloatArray(NUM_CLASSES) }
        interpreter?.run(batchInput, batchOutput)

        // Take first sample's predictions
        val predictions = batchOutput[0]

        // Calculate deltas for FL (output layer gradient)
        val deltas = FloatArray(NUM_CLASSES) { 0f }
        if (userProvidedLabelIndex != null && userProvidedLabelIndex in 0 until NUM_CLASSES) {
            for (i in 0 until NUM_CLASSES) {
                val target = if (i == userProvidedLabelIndex) 1.0f else 0.0f
                deltas[i] = predictions[i] - target
            }
        }
        return Pair(predictions, deltas)
    }

    fun close() { interpreter?.close() }
}