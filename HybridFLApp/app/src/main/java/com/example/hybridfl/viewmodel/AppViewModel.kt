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
            _flStatus.value = "Sending Weights to Server..."
            try {
                val response = RetrofitClient.apiService.sendWeights(
                    FLUpdateRequest(deviceId = deviceId, weightDeltas = deltas.toList())
                )
                if (response.isSuccessful) {
                    _flStatus.value = "FL Updated Successfully"
                } else {
                    _flStatus.value = "FL Update Failed"
                }
            } catch (e: Exception) {
                e.printStackTrace()
                _flStatus.value = "FL Network Error"
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        tfliteHelper.close()
    }
}
