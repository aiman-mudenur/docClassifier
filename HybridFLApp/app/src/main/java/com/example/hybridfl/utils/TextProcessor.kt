package com.example.hybridfl.utils

import android.content.Context
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Converts raw document text into a 100-dim float feature vector.
 *
 * Uses feature hashing (hashing trick) with unigrams + bigrams so that
 * every document produces DIFFERENT features regardless of vocabulary.
 * L2-normalised so values are always in a stable range for the TFLite model.
 */
class TextProcessor(private val context: Context) {

    companion object {
        const val FEATURE_DIM   = 100
        const val UNIGRAM_BINS  = 60   // first 60 dimensions = unigram TF
        const val BIGRAM_BINS   = 40   // last  40 dimensions = bigram TF
    }

    // Simple English stop-words to ignore
    private val stopWords = setOf(
        "the","a","an","is","it","in","on","at","to","of","and","or",
        "for","with","that","this","was","are","be","as","by","from",
        "has","have","had","not","but","we","they","he","she","you","i"
    )

    fun processText(text: String): FloatArray {
        val features = FloatArray(FEATURE_DIM) { 0f }

        if (text.isBlank()) return features

        // ── Tokenise ──────────────────────────────────────────────────────
        val words = text.lowercase()
            .replace(Regex("[^a-z0-9\\s]"), " ")
            .split(Regex("\\s+"))
            .filter { it.length > 2 && it !in stopWords }
            .take(500)          // cap at 500 words for speed

        if (words.isEmpty()) return features

        // ── Unigram TF (feature hashing into bins 0..59) ──────────────────
        for (word in words) {
            val bin = (word.hashCode() and 0x7FFF_FFFF) % UNIGRAM_BINS
            features[bin] += 1f
        }

        // ── Bigram TF (feature hashing into bins 60..99) ─────────────────
        for (i in 0 until words.size - 1) {
            val bigram = "${words[i]}_${words[i + 1]}"
            val bin = UNIGRAM_BINS + (bigram.hashCode() and 0x7FFF_FFFF) % BIGRAM_BINS
            features[bin] += 1f
        }

        // ── Sub-linear TF scaling: tf = 1 + log(tf) ──────────────────────
        for (i in features.indices) {
            if (features[i] > 0f) features[i] = 1f + ln(features[i]).toFloat()
        }

        // ── L2 normalise so the vector lives on the unit sphere ───────────
        val norm = sqrt(features.map { it * it }.sum())
        if (norm > 0f) for (i in features.indices) features[i] /= norm

        return features
    }
}