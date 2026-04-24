package com.yourapp.flclient;
 
import android.content.Context;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.util.Log;
 
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
 
import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
 
/**
 * FLClient — Federated Learning client for Android.
 *
 * Responsibilities:
 *  1. Register with FL server
 *  2. Pull global model weights
 *  3. Run local training (battery-aware epoch count)
 *  4. Upload updated weights with metadata
 *
 * Usage (from Activity/Service):
 *   FLClient client = new FLClient(context, "https://your-render-url.onrender.com");
 *   client.runFederatedRound(localSamples, topicCounts, docTypes, callback);
 */
public class FLClient {
 
    private static final String TAG = "FLClient";
 
    // ── Battery thresholds ────────────────────────────────────────────────────
    private static final int BATTERY_MIN_TO_TRAIN  = 20;   // % — skip if below
    private static final int BATTERY_FULL_EPOCHS   = 5;    // epochs when charged
    private static final int BATTERY_LOW_EPOCHS    = 2;    // epochs when low battery
    private static final int BATTERY_MID_THRESHOLD = 50;   // %
 
    private final Context context;
    private final String serverUrl;
    private final String deviceId;
    private final ExecutorService executor;
 
    // Current global weights (List of layers, each layer is List<List<Float>> or List<Float>)
    private List<Object> globalWeights = new ArrayList<>();
    private int currentRound = -1;
 
    public interface FLCallback {
        void onSuccess(String status, int round);
        void onError(String error);
    }
 
    public FLClient(Context context, String serverUrl) {
        this.context   = context.getApplicationContext();
        this.serverUrl = serverUrl;
        this.deviceId  = "android-" + UUID.randomUUID().toString().substring(0, 8);
        this.executor  = Executors.newSingleThreadExecutor();
    }
 
    // ── Public API ────────────────────────────────────────────────────────────
 
    /**
     * Full FL round: register → pull → train locally → upload.
     *
     * @param numSamples   Number of local training samples processed
     * @param topicCounts  Map of topicId → count (from local NLP)
     * @param docTypes     List of doc types ingested e.g. ["pdf","docx"]
     * @param localLoss    Loss from local training (0.0 if unknown)
     * @param callback     Result callback on main thread
     */
    public void runFederatedRound(
            final int numSamples,
            final Map<String, Integer> topicCounts,
            final List<String> docTypes,
            final float localLoss,
            final FLCallback callback) {
 
        executor.execute(() -> {
            try {
                // 1. Check battery before doing anything
                BatteryInfo battery = getBatteryInfo();
                Log.i(TAG, "Battery: " + battery.level + "% charging=" + battery.isCharging);
 
                if (!battery.isCharging && battery.level < BATTERY_MIN_TO_TRAIN) {
                    Log.w(TAG, "Battery too low (" + battery.level + "%). Skipping FL round.");
                    if (callback != null) callback.onError("Battery too low to train.");
                    return;
                }
 
                // 2. Register device
                registerDevice(battery);
 
                // 3. Pull latest global model
                boolean pulled = pullGlobalModel();
                if (!pulled) {
                    if (callback != null) callback.onError("Failed to pull global model.");
                    return;
                }
 
                // 4. Simulate local training (battery-aware epoch count)
                int epochs = chooseEpochs(battery);
                Log.i(TAG, "Local training: " + epochs + " epochs on " + numSamples + " samples");
                List<Object> updatedWeights = localTrain(globalWeights, epochs, numSamples);
 
                // 5. Upload updated weights
                JSONObject result = uploadWeights(
                        updatedWeights, numSamples, topicCounts, docTypes,
                        localLoss, battery);
 
                String status = result.optString("status", "unknown");
                int round     = result.optInt("round", currentRound);
                Log.i(TAG, "Upload result: " + status + " round=" + round);
 
                if (callback != null) callback.onSuccess(status, round);
 
            } catch (Exception e) {
                Log.e(TAG, "FL round failed: " + e.getMessage(), e);
                if (callback != null) callback.onError(e.getMessage());
            }
        });
    }
 
    // ── Step implementations ──────────────────────────────────────────────────
 
    private void registerDevice(BatteryInfo battery) {
        try {
            JSONObject body = new JSONObject();
            body.put("device_id",     deviceId);
            body.put("battery_level", battery.level);
            body.put("is_charging",   battery.isCharging);
            body.put("model_version", "1.0");
            postJson(serverUrl + "/register", body);
        } catch (Exception e) {
            Log.w(TAG, "Register failed (non-fatal): " + e.getMessage());
        }
    }
 
    private boolean pullGlobalModel() {
        try {
            String response = getRequest(serverUrl + "/get_global_model");
            JSONObject json = new JSONObject(response);
            currentRound  = json.optInt("round", 0);
            JSONArray arr = json.getJSONArray("weights");
            globalWeights = jsonArrayToList(arr);
            Log.i(TAG, "Pulled global model: round=" + currentRound
                    + " layers=" + globalWeights.size());
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Pull failed: " + e.getMessage(), e);
            return false;
        }
    }
 
    /**
     * Battery-aware epoch selection.
     * Low battery → fewer local epochs → less computation → less drain.
     */
    private int chooseEpochs(BatteryInfo battery) {
        if (battery.isCharging)            return BATTERY_FULL_EPOCHS;
        if (battery.level >= BATTERY_MID_THRESHOLD) return BATTERY_FULL_EPOCHS - 1;
        if (battery.level >= BATTERY_MIN_TO_TRAIN)  return BATTERY_LOW_EPOCHS;
        return 1;
    }
 
    /**
     * Simulated local training — in a real app replace this with TFLite
     * model training using the Android Task Library or LiteRT on-device training API.
     *
     * Here we add small Gaussian noise to represent a local gradient update.
     * The server-side FedAvg will average these across clients.
     */
    @SuppressWarnings("unchecked")
    private List<Object> localTrain(List<Object> weights, int epochs, int numSamples) {
        List<Object> updated = deepCopyWeights(weights);
        double lr = 0.001 * epochs;   // scale perturbation by epochs
 
        for (Object layer : updated) {
            if (layer instanceof List) {
                applyNoise((List<Object>) layer, lr);
            }
        }
        return updated;
    }
 
    @SuppressWarnings("unchecked")
    private void applyNoise(List<Object> layer, double scale) {
        for (int i = 0; i < layer.size(); i++) {
            Object elem = layer.get(i);
            if (elem instanceof List) {
                applyNoise((List<Object>) elem, scale);
            } else if (elem instanceof Number) {
                double val   = ((Number) elem).doubleValue();
                double noise = (Math.random() - 0.5) * 2 * scale;
                layer.set(i, (float) (val + noise));
            }
        }
    }
 
    private JSONObject uploadWeights(
            List<Object> weights,
            int numSamples,
            Map<String, Integer> topicCounts,
            List<String> docTypes,
            float localLoss,
            BatteryInfo battery) throws JSONException, IOException {
 
        JSONObject body = new JSONObject();
        body.put("device_id",     deviceId);
        body.put("weights",       listToJsonArray(weights));
        body.put("num_samples",   numSamples);
        body.put("battery_level", battery.level);
        body.put("is_charging",   battery.isCharging);
        body.put("local_loss",    localLoss);
        body.put("round",         currentRound);
 
        // Topic counts
        JSONObject topics = new JSONObject();
        for (Map.Entry<String, Integer> e : topicCounts.entrySet())
            topics.put(e.getKey(), e.getValue());
        body.put("topic_counts", topics);
 
        // Doc types
        JSONArray dtArr = new JSONArray();
        for (String dt : docTypes) dtArr.put(dt);
        body.put("doc_types", dtArr);
 
        String resp = postJson(serverUrl + "/upload_weights", body);
        return new JSONObject(resp);
    }
 
    // ── Battery helper ────────────────────────────────────────────────────────
 
    private BatteryInfo getBatteryInfo() {
        IntentFilter filter = new IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED);
        android.content.Intent intent = context.registerReceiver(null, filter);
        int level  = 100;
        boolean charging = true;
        if (intent != null) {
            int lvl   = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
            int scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
            level     = (scale > 0) ? (int) (lvl * 100f / scale) : 100;
            int status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1);
            charging  = status == BatteryManager.BATTERY_STATUS_CHARGING
                     || status == BatteryManager.BATTERY_STATUS_FULL;
        }
        return new BatteryInfo(level, charging);
    }
 
    static class BatteryInfo {
        final int level;
        final boolean isCharging;
        BatteryInfo(int level, boolean isCharging) {
            this.level = level; this.isCharging = isCharging;
        }
    }
 
    // ── Network helpers ───────────────────────────────────────────────────────
 
    private String getRequest(String urlStr) throws IOException {
        URL url = new URL(urlStr);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(15_000);
        conn.setReadTimeout(30_000);
        try (Scanner sc = new Scanner(conn.getInputStream(), "UTF-8")) {
            StringBuilder sb = new StringBuilder();
            while (sc.hasNextLine()) sb.append(sc.nextLine());
            return sb.toString();
        } finally {
            conn.disconnect();
        }
    }
 
    private String postJson(String urlStr, JSONObject body) throws IOException {
        URL url = new URL(urlStr);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json; utf-8");
        conn.setDoOutput(true);
        conn.setConnectTimeout(15_000);
        conn.setReadTimeout(60_000);   // weight upload may take longer
        try (OutputStream os = conn.getOutputStream()) {
            os.write(body.toString().getBytes(StandardCharsets.UTF_8));
        }
        int code = conn.getResponseCode();
        java.io.InputStream is = (code >= 200 && code < 300)
                ? conn.getInputStream() : conn.getErrorStream();
        try (Scanner sc = new Scanner(is, "UTF-8")) {
            StringBuilder sb = new StringBuilder();
            while (sc.hasNextLine()) sb.append(sc.nextLine());
            return sb.toString();
        } finally {
            conn.disconnect();
        }
    }
 
    // ── JSON ↔ List conversion ────────────────────────────────────────────────
 
    @SuppressWarnings("unchecked")
    private List<Object> jsonArrayToList(JSONArray arr) throws JSONException {
        List<Object> list = new ArrayList<>();
        for (int i = 0; i < arr.length(); i++) {
            Object elem = arr.get(i);
            if (elem instanceof JSONArray) {
                list.add(jsonArrayToList((JSONArray) elem));
            } else {
                list.add(((Number) elem).floatValue());
            }
        }
        return list;
    }
 
    @SuppressWarnings("unchecked")
    private JSONArray listToJsonArray(List<Object> list) throws JSONException {
        JSONArray arr = new JSONArray();
        for (Object elem : list) {
            if (elem instanceof List) {
                arr.put(listToJsonArray((List<Object>) elem));
            } else {
                arr.put(elem);
            }
        }
        return arr;
    }
 
    @SuppressWarnings("unchecked")
    private List<Object> deepCopyWeights(List<Object> src) {
        List<Object> copy = new ArrayList<>();
        for (Object elem : src) {
            if (elem instanceof List) {
                copy.add(deepCopyWeights((List<Object>) elem));
            } else {
                copy.add(elem);
            }
        }
        return copy;
    }
}
