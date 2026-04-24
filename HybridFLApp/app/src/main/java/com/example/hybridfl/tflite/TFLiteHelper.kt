package com.example.hybridfl.tflite

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class TFLiteHelper(private val context: Context) {
    private var interpreter: Interpreter? = null

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val assetFileDescriptor = context.assets.openFd("global_model.tflite")
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            val mappedByteBuffer: MappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            val options = Interpreter.Options()
            options.setNumThreads(2)
            interpreter = Interpreter(mappedByteBuffer, options)
        } catch (e: Exception) {
            e.printStackTrace()
            // Fast failure / debug fallback: Model might be missing in an empty layout
        }
    }

    // Returns a Pair of: <Probabilities FloatArray, Delta Weights FloatArray>
    fun runInferenceAndCalculateDeltas(inputFeatures: FloatArray, userProvidedLabelIndex: Int?): Pair<FloatArray, FloatArray> {
        val outputProbabilities = Array(1) { FloatArray(14) }
        
        interpreter?.run(arrayOf(inputFeatures), outputProbabilities)
        
        val predictions = outputProbabilities[0]
        val deltas = FloatArray(14) { 0f }
        
        // FedAvg Simulation:
        if (userProvidedLabelIndex != null && userProvidedLabelIndex in 0..13) {
            for (i in 0..13) {
                val target = if (i == userProvidedLabelIndex) 1.0f else 0.0f
                deltas[i] = predictions[i] - target 
            }
        }
        return Pair(predictions, deltas)
    }

    fun close() {
        interpreter?.close()
    }
}
