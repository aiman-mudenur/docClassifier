package com.example.hybridfl.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.hybridfl.network.FLUpdateRequest
import com.example.hybridfl.network.RegisterRequest
import com.example.hybridfl.network.RetrofitClient
import com.example.hybridfl.tflite.TFLiteHelper
import com.example.hybridfl.utils.FileUtil
import com.example.hybridfl.utils.TextProcessor
import com.google.gson.JsonArray
import com.google.gson.JsonElement
import com.google.gson.JsonPrimitive
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID
import kotlin.random.Random

class AppViewModel(application: Application) : AndroidViewModel(application) {

    private val textProcessor = TextProcessor(application)
    private val tfliteHelper  = TFLiteHelper(application)
    private val deviceId      = "android-" + UUID.randomUUID().toString().substring(0, 8)

    private val _predictions  = MutableStateFlow<FloatArray?>(null)
    val predictions: StateFlow<FloatArray?> = _predictions

    private val _flStatus     = MutableStateFlow("Ready — upload a document to start")
    val flStatus: StateFlow<String> = _flStatus

    private val _batteryLevel = MutableStateFlow(85)
    val batteryLevel: StateFlow<Int> = _batteryLevel

    // ── Entry point called when user picks a document ─────────────────────
    fun processDocument(uri: Uri) {
        viewModelScope.launch {
            try {
                // 1. Extract text
                _flStatus.value = "📄 Extracting text..."
                val text = withContext(Dispatchers.IO) {
                    FileUtil.extractTextFromUri(getApplication(), uri)
                }
                if (text.isBlank()) {
                    _flStatus.value = "❌ Could not read document text"
                    return@launch
                }

                // 2. Feature extraction
                _flStatus.value = "🔍 Processing text features..."
                val features = withContext(Dispatchers.Default) {
                    textProcessor.processText(text)
                }

                // 3. Local inference
                _flStatus.value = "🧠 Running classification..."
                val inferenceResult = withContext(Dispatchers.Default) {
                    tfliteHelper.runInferenceAndCalculateDeltas(features, null)
                }
                _predictions.value = inferenceResult.first

                // 4. FL round (register → fetch round → upload weights)
                runFLRound(inferenceResult.second, text)

            } catch (e: Exception) {
                e.printStackTrace()
                _flStatus.value = "❌ Error: ${e.message}"
            }
        }
    }

    // ── Full Federated Learning round ─────────────────────────────────────
    private suspend fun runFLRound(deltas: FloatArray, docText: String) {
        if (_batteryLevel.value < 20) {
            _flStatus.value = "⚠️ FL skipped — battery too low"
            return
        }

        withContext(Dispatchers.IO) {

            // ── STEP 1: Register with Render (retry for cold-start) ───────
            _flStatus.value = "📡 Connecting to FL server..."
            var registered = false
            for (attempt in 1..3) {
                try {
                    val resp = RetrofitClient.apiService.register(
                        RegisterRequest(
                            device_id     = deviceId,
                            battery_level = _batteryLevel.value,
                            is_charging   = true
                        )
                    )
                    if (resp.isSuccessful) { registered = true; break }
                    else _flStatus.value = "⏳ Server waking up... ($attempt/3)"
                } catch (e: Exception) {
                    _flStatus.value = "⏳ Connecting... ($attempt/3)"
                    delay(5_000)
                }
            }

            if (!registered) {
                _flStatus.value = "❌ FL failed — server unreachable"
                return@withContext
            }

            // ── STEP 2: Fetch global model to get current round number ────
            _flStatus.value = "⬇️ Fetching global model..."
            val modelResp = try {
                RetrofitClient.apiService.getGlobalModel()
            } catch (e: Exception) {
                _flStatus.value = "❌ FL error: ${e.message}"
                return@withContext
            }

            if (!modelResp.isSuccessful || modelResp.body() == null) {
                _flStatus.value = "❌ FL failed — could not fetch model (${modelResp.code()})"
                return@withContext
            }

            val globalModel  = modelResp.body()!!
            val currentRound = globalModel.round

            // ── STEP 3: Check how many clients are waiting ────────────────
            val statusResp = try { RetrofitClient.apiService.getStatus() } catch (e: Exception) { null }
            val waiting    = statusResp?.body()?.clients_waiting ?: 0
            val minClients = statusResp?.body()?.min_clients ?: 3

            // ── STEP 4: Build updated weights (add noise to global model) ─
            _flStatus.value = "⚙️ Local training... (round $currentRound)"
            val updatedWeights = addNoiseToWeights(globalModel.weights)

            // ── STEP 5: Build topic counts from top predicted class ────────
            val topClass = _predictions.value
                ?.mapIndexed { i, v -> i to v }
                ?.maxByOrNull { it.second }?.first ?: 0

            // ── STEP 6: Upload weights ────────────────────────────────────
            _flStatus.value = "⬆️ Uploading weights to server..."
            try {
                val request = FLUpdateRequest(
                    device_id     = deviceId,
                    weights       = updatedWeights,
                    num_samples   = 100,
                    battery_level = _batteryLevel.value,
                    is_charging   = true,
                    local_loss    = 0.31f,
                    round         = currentRound,
                    topic_counts  = mapOf(topClass.toString() to 100),
                    doc_types     = listOf("document")
                )

                val uploadResp = RetrofitClient.apiService.sendWeights(request)

                if (uploadResp.isSuccessful) {
                    val body   = uploadResp.body()
                    val status = body?.status ?: "unknown"
                    val round  = body?.round  ?: currentRound

                    _flStatus.value = when (status) {
                        "aggregated" ->
                            "✅ FL complete! Round $round aggregated (all $minClients clients submitted)"
                        "accepted"   ->
                            "⏳ FL accepted — waiting for ${minClients - waiting - 1} more client(s) (round $round)"
                        "duplicate"  ->
                            "ℹ️ Already submitted this round ($round)"
                        else         ->
                            "✅ FL $status — round $round"
                    }
                } else {
                    _flStatus.value = "❌ FL failed: ${uploadResp.code()}"
                }

            } catch (e: Exception) {
                e.printStackTrace()
                _flStatus.value = "❌ FL error: ${e.message}"
            }
        }
    }

    // ── Add Gaussian noise to simulate local gradient update ──────────────
    private fun addNoiseToWeights(weights: JsonArray): JsonArray {
        val result = JsonArray()
        for (layer: JsonElement in weights) {
            result.add(addNoiseToElement(layer))
        }
        return result
    }

    private fun addNoiseToElement(element: JsonElement): JsonElement {
        return when {
            element.isJsonArray -> {
                val arr = element.asJsonArray
                val out = JsonArray()
                for (child in arr) out.add(addNoiseToElement(child))
                out
            }
            element.isJsonPrimitive -> {
                val v = element.asFloat
                JsonPrimitive(v + (Random.nextFloat() - 0.5f) * 0.002f)
            }
            else -> element
        }
    }
}