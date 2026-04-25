package com.example.hybridfl.viewmodel

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.hybridfl.network.FLUpdateRequest
import com.example.hybridfl.network.RetrofitClient
import com.example.hybridfl.tflite.TFLiteHelper
import com.example.hybridfl.utils.FileUtil
import com.example.hybridfl.utils.TextProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.UUID

class AppViewModel(application: Application) : AndroidViewModel(application) {

    private val textProcessor = TextProcessor(application)
    private val tfliteHelper = TFLiteHelper(application)
    private val deviceId = UUID.randomUUID().toString()

    private val _predictions = MutableStateFlow<FloatArray?>(null)
    val predictions: StateFlow<FloatArray?> = _predictions

    private val _flStatus = MutableStateFlow("Waiting")
    val flStatus: StateFlow<String> = _flStatus

    private val _batteryLevel = MutableStateFlow(85) // Simulated battery
    val batteryLevel: StateFlow<Int> = _batteryLevel

    fun processDocument(uri: Uri) {
        viewModelScope.launch {
            _flStatus.value = "Extracting text..."
            
            val text = withContext(Dispatchers.IO) {
                FileUtil.extractTextFromUri(getApplication(), uri)
            }

            _flStatus.value = "Processing NLP..."
            val features = textProcessor.processText(text)

            _flStatus.value = "Running Inference..."
            
            // Assume system defaults label to 3 as representation or via UI dropdown
            val simulatedUserLabel = 3 

            val result = withContext(Dispatchers.Default) {
                tfliteHelper.runInferenceAndCalculateDeltas(features, simulatedUserLabel)
            }

            _predictions.value = result.first
            _flStatus.value = "Calculating FL Deltas..."

            sendFederatedUpdate(result.second)
        }
    }

    private fun sendFederatedUpdate(deltas: FloatArray) {
        if (_batteryLevel.value < 20) {
            _flStatus.value = "FL Skipped (Low Battery)"
            return
        }

        viewModelScope.launch {
            _flStatus.value = "Sending weights to server..."
            try {
                // Convert flat delta array into 6 layers matching model architecture:
                // W1(128×64), b1(64), W2(64×32), b2(32), W3(32×5), b3(5)
                val layerSizes = listOf(128*64, 64, 64*32, 32, 32*5, 5)
                val layerShapes = listOf(
                    Pair(128, 64), null, Pair(64, 32), null, Pair(32, 5), null
                )
                val layers = mutableListOf<List<Float>>()
                var offset = 0
                for (size in layerSizes) {
                    val slice = deltas.slice(offset until (offset + size).coerceAtMost(deltas.size))
                    layers.add(slice)
                    offset += size
                }

                val request = FLUpdateRequest(
                    device_id = deviceId,
                    weights = layers,
                    num_samples = 100,
                    battery_level = _batteryLevel.value,
                    is_charging = false,
                    local_loss = 0.31f,
                    round = 0,
                    topic_counts = mapOf("0" to 50, "1" to 30, "2" to 20),
                    doc_types = listOf("pdf", "docx")
                )

                val response = RetrofitClient.apiService.sendWeights(request)
                if (response.isSuccessful) {
                    val body = response.body()
                    _flStatus.value = "FL ${body?.status ?: "done"} — round ${body?.round ?: 0}"
                } else {
                    _flStatus.value = "FL failed: ${response.code()}"
                }
            } catch (e: Exception) {
                e.printStackTrace()
                _flStatus.value = "FL error: ${e.message}"
            }
        }
    }