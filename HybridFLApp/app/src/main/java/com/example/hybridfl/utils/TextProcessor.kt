package com.example.hybridfl.utils

import android.content.Context
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader

class TextProcessor(private val context: Context) {
    private val vocab = mutableMapOf<String, Int>()
    private val maxLen = 100

    init {
        loadVocab()
    }

    private fun loadVocab() {
        try {
            // Load a lightweight vocab json (simulation of WordPiece/BPE)
            val inputStream = context.assets.open("vocab.json")
            val br = BufferedReader(InputStreamReader(inputStream))
            val jsonText = br.use { it.readText() }
            val jsonObject = JSONObject(jsonText)
            
            jsonObject.keys().forEach { key ->
                vocab[key] = jsonObject.getInt(key)
            }
        } catch (e: Exception) {
            e.printStackTrace()
            // Fast fail fallback
            vocab["default"] = 1
        }
    }

    fun processText(text: String): FloatArray {
        // We use FloatArray representing the INT IDs because TFLite often maps better to Floats for standard MLPs.
        val words = text.lowercase().replace(Regex("[^a-z0-9 ]"), " ").split("\\s+".toRegex())
        val sequence = FloatArray(maxLen) { 0f } // 0 corresponds to <PAD> or <UNK>

        var index = 0
        for (word in words) {
            if (word.isBlank()) continue
            if (index >= maxLen) break
            sequence[index] = (vocab[word] ?: 1).toFloat() // Assume 1 is <UNK>
            index++
        }
        return sequence
    }
}
