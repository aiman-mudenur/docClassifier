package com.yourapp.flclient;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * TFLiteClassifier
 * ----------------
 * Loads model.tflite from assets and runs inference + local training.
 *
 * Add to build.gradle (app):
 *   implementation 'org.tensorflow:tensorflow-lite:2.14.0'
 *   implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
 *
 * Place your model.tflite in:   app/src/main/assets/model.tflite
 *
 * Input:  float[1][128]  — feature vector from DocumentProcessor.toFeatureVector()
 * Output: float[1][5]    — class probabilities for 5 topics
 */
public class TFLiteClassifier {

    private static final String TAG        = "TFLiteClassifier";
    private static final String MODEL_FILE = "model.tflite";
    private static final int    INPUT_DIM  = 128;
    private static final int    NUM_CLASSES = 5;

    private Interpreter interpreter;
    private final Context context;

    public TFLiteClassifier(Context context) {
        this.context = context.getApplicationContext();
        try {
            Interpreter.Options opts = new Interpreter.Options();
            opts.setNumThreads(2);
            interpreter = new Interpreter(loadModelFile(), opts);
            Log.i(TAG, "TFLite model loaded successfully.");
        } catch (Exception e) {
            Log.e(TAG, "Failed to load TFLite model: " + e.getMessage(), e);
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /**
     * Classify a feature vector.
     * @param featureVec float[128] from DocumentProcessor.toFeatureVector()
     * @return TopicResult with predicted class index and probabilities
     */
    public TopicResult classify(float[] featureVec) {
        if (interpreter == null) return new TopicResult(-1, new float[NUM_CLASSES]);

        float[][] input  = new float[1][INPUT_DIM];
        float[][] output = new float[1][NUM_CLASSES];
        System.arraycopy(featureVec, 0, input[0], 0, Math.min(featureVec.length, INPUT_DIM));

        interpreter.run(input, output);

        float[] probs   = output[0];
        int bestClass   = 0;
        for (int i = 1; i < NUM_CLASSES; i++)
            if (probs[i] > probs[bestClass]) bestClass = i;

        Log.d(TAG, "Predicted class=" + bestClass + " conf=" + probs[bestClass]);
        return new TopicResult(bestClass, probs);
    }

    /**
     * Classify a list of documents and return a topic count map.
     * This is what gets sent to the FL server as topic_counts.
     */
    public Map<String, Integer> classifyBatch(List<float[]> featureVectors) {
        Map<String, Integer> counts = new HashMap<>();
        for (float[] vec : featureVectors) {
            TopicResult result = classify(vec);
            if (result.classIndex >= 0) {
                String key = String.valueOf(result.classIndex);
                counts.put(key, counts.getOrDefault(key, 0) + 1);
            }
        }
        return counts;
    }

    /**
     * Percentage breakdown of topics across a batch.
     */
    public Map<String, Float> topicPercentages(List<float[]> featureVectors) {
        Map<String, Integer> counts = classifyBatch(featureVectors);
        int total = featureVectors.size();
        Map<String, Float> pct = new HashMap<>();
        for (Map.Entry<String, Integer> e : counts.entrySet())
            pct.put(e.getKey(), e.getValue() * 100f / total);
        return pct;
    }

    // ── Weight extraction for FL ──────────────────────────────────────────────

    /**
     * Extract current interpreter weights as a nested float list.
     * Used to send to the FL server after local training.
     *
     * Note: Full on-device training (gradient updates) requires TFLite
     * Model Maker or the experimental on-device training API.
     * For the FL simulation, we extract the static weights and the
     * FLClient.localTrain() method adds simulated gradient noise.
     *
     * To extract real gradients from your model, use:
     *   org.tensorflow.lite.support.model.Model with trainable layers.
     */
    public float[][][] extractWeights() {
        if (interpreter == null) return new float[0][][];
        // Placeholder — returns empty shell; replace with real extraction
        // using interpreter.getSignatureRunner() for trained models.
        Log.w(TAG, "extractWeights(): using simulated weights. "
                + "Integrate real gradient extraction for production.");
        return new float[0][][];
    }

    // ── Model loading ─────────────────────────────────────────────────────────

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor afd = context.getAssets().openFd(MODEL_FILE);
        FileInputStream fis = new FileInputStream(afd.getFileDescriptor());
        FileChannel channel = fis.getChannel();
        return channel.map(FileChannel.MapMode.READ_ONLY,
                afd.getStartOffset(), afd.getDeclaredLength());
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }

    // ── Data classes ──────────────────────────────────────────────────────────

    public static class TopicResult {
        public final int     classIndex;
        public final float[] probabilities;
        public final String  label;

        private static final String[] LABELS = {
            "Finance", "Health", "Technology", "Legal", "Education"
        };

        TopicResult(int classIndex, float[] probabilities) {
            this.classIndex     = classIndex;
            this.probabilities  = probabilities;
            this.label          = (classIndex >= 0 && classIndex < LABELS.length)
                                  ? LABELS[classIndex] : "Unknown";
        }

        public float confidence() {
            return (classIndex >= 0) ? probabilities[classIndex] : 0f;
        }

        @Override
        public String toString() {
            return label + " (" + String.format("%.1f", confidence() * 100) + "%)";
        }
    }
}
