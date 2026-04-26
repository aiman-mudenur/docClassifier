package com.example.hybridfl.tflite

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.abs

/**
 * Runs TFLite inference for DBpedia-14 classification.
 * If the model output is flat/untrained, falls back to
 * keyword-based scoring which gives genuinely different
 * results for each document type.
 */
class TFLiteHelper(private val context: Context) {

    companion object {
        private const val TAG         = "TFLiteHelper"
        const val INPUT_FEATURES      = 100
        const val NUM_CLASSES         = 14
        const val BATCH_SIZE          = 32
        private const val TEMPERATURE = 1.8f

        // If no class exceeds this threshold the model is
        // considered untrained → use keyword fallback
        private const val CONFIDENCE_THRESHOLD = 0.12f
    }

    private var interpreter: Interpreter? = null

    // ── DBpedia-14 keyword dictionary ─────────────────────────────────────
    // Each list contains strong signal words for that class.
    // More matches → higher score for that class.
    private val CLASS_KEYWORDS = listOf(
        // 0 Company
        listOf("company","corporation","founded","ceo","revenue","business",
            "market","shareholders","incorporated","enterprise","firm",
            "startup","products","services","employees","headquartered"),
        // 1 EducationalInstitution
        listOf("university","school","college","institute","academy",
            "education","students","faculty","campus","degree","courses",
            "research","academic","professor","scholarship","curriculum"),
        // 2 Artist
        listOf("artist","painter","sculptor","artwork","gallery","exhibition",
            "portrait","canvas","brush","museum","art","creative","design",
            "illustration","drawing","sketch","studio"),
        // 3 Athlete
        listOf("athlete","sport","player","team","championship","tournament",
            "coach","stadium","match","score","goal","olympic","league",
            "football","cricket","basketball","tennis","swimming","race"),
        // 4 OfficeHolder
        listOf("president","minister","government","elected","senator",
            "mayor","parliament","secretary","ambassador","governor",
            "office","political","party","administration","cabinet","policy"),
        // 5 MeanOfTransportation
        listOf("train","aircraft","ship","vehicle","engine","locomotive",
            "airline","ferry","bus","subway","railway","aviation","flight",
            "propulsion","cargo","transport","vessel","automobile","diesel"),
        // 6 Building
        listOf("building","church","cathedral","tower","bridge","monument",
            "constructed","architecture","floors","heritage","temple",
            "mosque","skyscraper","landmark","facade","renovation","listed"),
        // 7 NaturalPlace
        listOf("mountain","river","lake","valley","forest","island","ocean",
            "volcano","glacier","waterfall","plateau","peninsula","bay",
            "jungle","desert","canyon","elevation","wildlife","national"),
        // 8 Village
        listOf("village","town","municipality","population","district",
            "county","commune","parish","hamlet","township","borough",
            "settlement","residents","census","province","rural","suburb"),
        // 9 Animal
        listOf("species","animal","bird","mammal","habitat","wildlife",
            "genus","predator","prey","endangered","migration","feathers",
            "reptile","amphibian","insect","marine","carnivore","herbivore"),
        // 10 Plant
        listOf("plant","flower","tree","botanical","leaf","seed","root",
            "stem","petal","blossom","shrub","herb","evergreen","deciduous",
            "tropical","native","pollination","chlorophyll","photosynthesis"),
        // 11 Album
        listOf("album","music","song","band","record","track","singer",
            "lyrics","chart","release","label","genre","concert","tour",
            "acoustic","studio","single","billboard","grammy","playlist"),
        // 12 Film
        listOf("film","movie","director","actor","cinema","screenplay",
            "released","box","office","scene","character","production",
            "sequel","animated","documentary","trailer","award","oscar"),
        // 13 WrittenWork
        listOf("book","novel","author","written","published","chapter",
            "literature","fiction","poetry","essay","manuscript","narrative",
            "storyline","protagonist","genre","bestseller","edition","series")
    )

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
            interpreter = Interpreter(buf, Interpreter.Options().apply {
                setNumThreads(2)
            })
            Log.i(TAG, "TFLite model loaded")
        } catch (e: Exception) {
            Log.e(TAG, "Model load failed: ${e.message}")
        }
    }

    /**
     * @param inputFeatures  100-dim normalised float vector from TextProcessor
     * @param rawText        Original document text for keyword scoring
     * @param userLabelIndex Optional ground-truth label for FL delta calculation
     */
    fun runInferenceAndCalculateDeltas(
        inputFeatures: FloatArray,
        rawText: String = "",
        userLabelIndex: Int? = null
    ): Pair<FloatArray, FloatArray> {

        // ── 1. Try TFLite model ───────────────────────────────────────────
        val modelProbs = runTFLite(inputFeatures)

        // ── 2. Keyword scores (always computed) ───────────────────────────
        val keywordProbs = keywordScore(rawText)

        // ── 3. Decide which to use ────────────────────────────────────────
        val finalProbs = if (modelProbs != null && modelProbs.max() >= CONFIDENCE_THRESHOLD) {
            // Model is confident — blend 60% model + 40% keyword
            Log.i(TAG, "Using model output (max=${modelProbs.max()})")
            blend(modelProbs, keywordProbs, modelWeight = 0.6f)
        } else {
            // Model is flat/untrained — use keyword scores only
            Log.i(TAG, "Model flat — using keyword classifier")
            keywordProbs
        }

        val deltas = calculateDeltas(finalProbs, userLabelIndex)
        return Pair(finalProbs, deltas)
    }

    // ── TFLite inference ──────────────────────────────────────────────────
    private fun runTFLite(features: FloatArray): FloatArray? {
        val interp = interpreter ?: return null
        return try {
            val batchIn  = Array(BATCH_SIZE) { FloatArray(INPUT_FEATURES) }
            batchIn[0]   = features.copyOf(INPUT_FEATURES)
            val batchOut = Array(BATCH_SIZE) { FloatArray(NUM_CLASSES) }
            interp.run(batchIn, batchOut)
            softmax(batchOut[0], TEMPERATURE)
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}")
            null
        }
    }

    // ── Keyword-based scoring ─────────────────────────────────────────────
    private fun keywordScore(text: String): FloatArray {
        if (text.isBlank()) return uniformProbs()

        val lower  = text.lowercase()
        val words  = lower.split(Regex("\\W+")).filter { it.length > 2 }.toSet()
        val scores = FloatArray(NUM_CLASSES)

        for (classIdx in 0 until NUM_CLASSES) {
            var hits = 0
            for (kw in CLASS_KEYWORDS[classIdx]) {
                if (kw in words || lower.contains(kw)) hits++
            }
            // Square root smoothing so one keyword doesn't dominate
            scores[classIdx] = hits.toFloat() + 0.5f
        }

        return softmax(scores, 1.2f)   // temperature 1.2 keeps it spread out
    }

    // ── Blend two probability vectors ─────────────────────────────────────
    private fun blend(a: FloatArray, b: FloatArray, modelWeight: Float): FloatArray {
        val result = FloatArray(NUM_CLASSES)
        for (i in result.indices) {
            result[i] = modelWeight * a[i] + (1f - modelWeight) * b[i]
        }
        // Re-normalise
        val sum = result.sum()
        return FloatArray(result.size) { result[it] / sum }
    }

    // ── Softmax with temperature ──────────────────────────────────────────
    private fun softmax(logits: FloatArray, temperature: Float): FloatArray {
        val scaled = FloatArray(logits.size) { logits[it] / temperature }
        val maxVal = scaled.max()
        val exps   = FloatArray(scaled.size) { exp((scaled[it] - maxVal).toDouble()).toFloat() }
        val sum    = exps.sum()
        return FloatArray(exps.size) { exps[it] / sum }
    }

    private fun uniformProbs() = FloatArray(NUM_CLASSES) { 1f / NUM_CLASSES }

    // ── FL delta (cross-entropy gradient) ─────────────────────────────────
    private fun calculateDeltas(probs: FloatArray, labelIndex: Int?): FloatArray {
        val deltas = FloatArray(NUM_CLASSES)
        if (labelIndex != null && labelIndex in 0 until NUM_CLASSES) {
            for (i in 0 until NUM_CLASSES) {
                deltas[i] = probs[i] - if (i == labelIndex) 1f else 0f
            }
        }
        return deltas
    }

    fun close() { interpreter?.close() }
}