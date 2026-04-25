package com.example.hybridfl.network

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

data class FLUpdateRequest(
    val device_id: String,
    val weights: List<List<Float>>,   // matches app.py: list of layers
    val num_samples: Int,
    val battery_level: Int,
    val is_charging: Boolean,
    val local_loss: Float,
    val round: Int,
    val topic_counts: Map<String, Int>,
    val doc_types: List<String>
)

data class FLUpdateResponse(
    val status: String,
    val message: String?,
    val round: Int?,
    val avg_loss: Float?,
    val topic_pct: Map<String, Float>?
)

data class GlobalModelResponse(
    val weights: List<List<Float>>,
    val round: Int,
    val layers: Int
)

interface ApiService {

    @POST("/upload_weights")
    suspend fun sendWeights(@Body request: FLUpdateRequest): Response<FLUpdateResponse>

    @GET("/get_global_model")
    suspend fun getGlobalModel(): Response<GlobalModelResponse>

    @GET("/health")
    suspend fun health(): Response<Map<String, Any>>
}