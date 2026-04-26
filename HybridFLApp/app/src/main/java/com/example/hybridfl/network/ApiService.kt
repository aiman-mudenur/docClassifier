package com.example.hybridfl.network

import com.google.gson.JsonArray
import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

data class RegisterRequest(
    val device_id: String,
    val battery_level: Int,
    val is_charging: Boolean,
    val model_version: String = "1.0"
)

data class RegisterResponse(
    val status: String,
    val device_id: String
)

data class FLUpdateRequest(
    val device_id: String,
    val weights: JsonArray,
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
    val round: Int?
)

data class GlobalModelResponse(
    val weights: JsonArray,
    val round: Int,
    val layers: Int,
    val num_classes: Int
)

data class StatusResponse(
    val round: Int,
    val clients_waiting: Int,
    val min_clients: Int,
    val registered: Int
)

interface ApiService {

    @POST("/register")
    suspend fun register(@Body request: RegisterRequest): Response<RegisterResponse>

    @GET("/get_global_model")
    suspend fun getGlobalModel(): Response<GlobalModelResponse>

    @POST("/upload_weights")
    suspend fun sendWeights(@Body request: FLUpdateRequest): Response<FLUpdateResponse>

    @GET("/status")
    suspend fun getStatus(): Response<StatusResponse>

    @GET("/health")
    suspend fun health(): Response<Map<String, Any>>
}