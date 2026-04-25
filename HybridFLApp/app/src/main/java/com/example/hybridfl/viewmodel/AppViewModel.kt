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

    private val _predictions = MutableStateFlow<FloatArray?>(null)
    val predictions: StateFlow<FloatArray?> = _predictions

    private val _flStatus = MutableStateFlow("Waiting")
    val flStatus: StateFlow<String> = _flStatus

    private val _batteryLevel = MutableStateFlow(85)
    val batteryLevel: StateFlow<Int> = _batteryLevel

    // ── Main entry point ──────────────────────────────────────────────────
    fun processDocument(uri: Uri) {
        viewModelScope.launch {
            try {
                _flStatus.value = "Extracting text..."
                val text = withContext(Dispatchers.IO) {
                    FileUtil.extractTextFromUri(getApplication(), uri)
                }

                _flStatus.value = "Processing NLP..."
                val features = textProcessor.processText(text)

                _flStatus.value = "Running inference..."
                val result = withContext(Dispatchers.Default) {
                    tfliteHelper.runInferenceAndCalculateDeltas(features, 3)
                }
                _predictions.value = result.first

                // Now do the full FL round
                runFLRound()

            } catch (e: Exception) {
                e.printStackTrace()
                _flStatus.value = "Error: ${e.message}"
            }
        }
    }

    // ── FL Round: register → fetch model → train → upload ─────────────────
    private suspend fun runFLRound() {
        if (_batteryLevel.value < 20) {
            _flStatus.value = "FL Skipped (Low Battery)"
            return
        }

        withContext(Dispatchers.IO) {

            // STEP 1 — Register with retry (handles Render cold-start)
            _flStatus.value = "Registering device..."
            var registered = false
            for (attempt in 1..3) {
                try {
                    val regResp = RetrofitClient.apiService.register(
                        RegisterRequest(
                            device_id     = deviceId,
                            battery_level = _batteryLevel.value,
                            is_charging   = true
                        )
                    )
                    if (regResp.isSuccessful) {
                        registered = true
                        break
                    }
                } catch (e: Exception) {
                    _flStatus.value = "Connecting to server (attempt $attempt/3)..."
                    delay(4_000)
                }
            }
            if (!registered) {
                _flStatus.value = "FL failed: server unreachable"
                return@withContext
            }

            // STEP 2 — Fetch global model to get current weights + round number
            _flStatus.value = "Fetching global model..."
            val modelResp = try {
                RetrofitClient.apiService.getGlobalModel()
            } catch (e: Exception) {
                _flStatus.value = "FL error: ${e.message}"
                return@withContext
            }

            if (!modelResp.isSuccessful || modelResp.body() == null) {
                _flStatus.value = "FL failed: could not fetch model"
                return@withContext
            }

            val globalModel = modelResp.body()!!
            val currentRound = globalModel.round

            // STEP 3 — Simulate local training: add tiny noise to each weight
            _flStatus.value = "Local training..."
            val updatedWeights = addNoiseToWeights(globalModel.weights)

            // STEP 4 — Upload updated weights
            _flStatus.value = "Sending weights to server..."
            try {
                val request = FLUpdateRequest(
                    device_id     = deviceId,
                    weights       = updatedWeights,
                    num_samples   = 100,
                    battery_level = _batteryLevel.value,
                    is_charging   = true,
                    local_loss    = 0.31f,
                    round         = currentRound,
                    topic_counts  = mapOf("0" to 50, "1" to 30, "2" to 20),
                    doc_types     = listOf("pdf", "docx")
                )

                val response = RetrofitClient.apiService.sendWeights(request)
                if (response.isSuccessful) {
                    val body = response.body()
                    val status = body?.status ?: "done"
                    val round  = body?.round  ?: currentRound
                    // accepted = waiting for more clients
                    // aggregated = all 3 clients submitted, FL round complete
                    _flStatus.value = when (status) {
                        "aggregated" -> "FL complete — round $round done ✓"
                        "accepted"   -> "FL accepted — waiting for other clients (round $round)"
                        "duplicate"  -> "FL duplicate — already submitted this round"
                        else         -> "FL $status — round $round"
                    }
                } else {
                    _flStatus.value = "FL failed: ${response.code()}"
                }
            } catch (e: Exception) {
                e.printStackTrace()
                _flStatus.value = "FL error: ${e.message}"
            }
        }
    }

    // ── Add small Gaussian noise to simulate local gradient update ─────────
    // Preserves the exact nested structure (2D matrices stay 2D, 1D biases stay 1D)
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
                val arr    = element.asJsonArray
                val result = JsonArray()
                for (child in arr) result.add(addNoiseToElement(child))
                result
            }
            element.isJsonPrimitive -> {
                // Add tiny noise: ±0.001
                val original = element.asFloat
                val noisy    = original + (Random.nextFloat() - 0.5f) * 0.002f
                com.google.gson.JsonPrimitive(noisy)
            }
            else -> element
        }
    }
}