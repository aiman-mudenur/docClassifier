package com.example.hybridfl.network

import retrofit2.Response
import retrofit2.http.Body
import retrofit2.http.POST

data class FLUpdateRequest(
    val deviceId: String,
    val weightDeltas: List<Float>
)

data class FLUpdateResponse(
    val status: String,
    val message: String
)

interface ApiService {
    @POST("/api/fl/aggregate")
    suspend fun sendWeights(@Body updateRequest: FLUpdateRequest): Response<FLUpdateResponse>
}
