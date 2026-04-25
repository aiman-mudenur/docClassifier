package com.example.hybridfl

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.hybridfl.viewmodel.AppViewModel

// ✅ DBpedia-14 class names
val CLASS_NAMES = listOf(
    "Company",
    "EducationalInstitution",
    "Artist",
    "Athlete",
    "OfficeHolder",
    "MeanOfTransportation",
    "Building",
    "NaturalPlace",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "WrittenWork"
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
fun HybridFLScreen(viewModel: AppViewModel = viewModel()) {
    val predictions by viewModel.predictions.collectAsState()
    val flStatus by viewModel.flStatus.collectAsState()
    val battery by viewModel.batteryLevel.collectAsState()

    val filePickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.processDocument(it) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        Text(
            text = "Privacy-Preserving Document Classification",
            style = MaterialTheme.typography.titleLarge
        )

        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Device Resources")
                Text("Simulated Battery: $battery%")
                Text("Compute Available: High")
            }
        }

        Button(
            onClick = { filePickerLauncher.launch("*/*") },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Upload Document")
        }

        Text(
            text = "FL Status: $flStatus",
            style = MaterialTheme.typography.bodyLarge,
            color = MaterialTheme.colorScheme.primary
        )

        predictions?.let { probs ->
            Text(
                text = "Classification Results:",
                style = MaterialTheme.typography.titleMedium
            )

            // ✅ Show Top 5 with real DBpedia class names
            val top5 = probs.mapIndexed { index, prob -> Pair(index, prob) }
                .sortedByDescending { it.second }
                .take(5)

            top5.forEach { (index, prob) ->
                val percentage = (prob * 100).toInt()
                val className = CLASS_NAMES.getOrElse(index) { "Class $index" }
                Text("$className: $percentage%")
            }
        }
    }
}