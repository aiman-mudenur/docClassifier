package com.example.hybridfl

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.hybridfl.viewmodel.AppViewModel

val CLASS_NAMES = listOf(
    "Company", "EducationalInstitution", "Artist", "Athlete",
    "OfficeHolder", "MeanOfTransportation", "Building", "NaturalPlace",
    "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        com.example.hybridfl.utils.FileUtil.initPdfBox(this)
        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    HybridFLScreen()
                }
            }
        }
    }
}

@Composable
fun HybridFLScreen(vm: AppViewModel = viewModel()) {
    val predictions by vm.predictions.collectAsState()
    val flStatus    by vm.flStatus.collectAsState()
    val battery     by vm.batteryLevel.collectAsState()

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? -> uri?.let { vm.processDocument(it) } }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // ── Title ─────────────────────────────────────────────────────────
        Text(
            "Privacy-Preserving Document Classification",
            style = MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold
        )

        // ── Device info card ──────────────────────────────────────────────
        Card(
            modifier  = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(4.dp),
            colors    = CardDefaults.cardColors(containerColor = Color(0xFFF0EEF8))
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Device Resources", fontWeight = FontWeight.SemiBold)
                Text("Simulated Battery: $battery%")
                Text("Compute Available: High")
            }
        }

        // ── Upload button ─────────────────────────────────────────────────
        Button(
            onClick   = { launcher.launch("*/*") },
            modifier  = Modifier.fillMaxWidth().height(52.dp),
            shape     = RoundedCornerShape(12.dp),
            colors    = ButtonDefaults.buttonColors(containerColor = Color(0xFF6200EE))
        ) {
            Text("Upload Document", fontSize = 16.sp, fontWeight = FontWeight.Bold)
        }

        // ── FL Status ─────────────────────────────────────────────────────
        Text(
            text  = "FL Status: $flStatus",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.primary
        )

        // ── Classification Results ────────────────────────────────────────
        predictions?.let { probs ->
            Text(
                "Classification Results:",
                style      = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            // Top 5 classes sorted by probability
            val top5 = probs
                .mapIndexed { idx, prob -> idx to prob }
                .sortedByDescending { it.second }
                .take(5)

            top5.forEach { (idx, prob) ->
                val className = CLASS_NAMES.getOrElse(idx) { "Class $idx" }
                val pct       = (prob * 100).toInt()
                val barWidth  = prob.coerceIn(0f, 1f)

                Column(modifier = Modifier.fillMaxWidth()) {
                    Row(
                        modifier              = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment     = Alignment.CenterVertically
                    ) {
                        Text(className, fontWeight = FontWeight.Medium)
                        Text("$pct%", fontWeight = FontWeight.Bold,
                            color = if (idx == top5[0].first) Color(0xFF6200EE) else Color.Gray)
                    }
                    Spacer(modifier = Modifier.height(4.dp))
                    // Progress bar
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(8.dp)
                            .background(Color(0xFFE0E0E0), RoundedCornerShape(4.dp))
                    ) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth(barWidth)
                                .height(8.dp)
                                .background(
                                    if (idx == top5[0].first) Color(0xFF6200EE)
                                    else Color(0xFFBBAADD),
                                    RoundedCornerShape(4.dp)
                                )
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }
            }
        }
    }
}