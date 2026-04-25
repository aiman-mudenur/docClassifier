package com.yourapp.flclient;

import android.content.Context;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class FLClient {

    private static final String TAG = "FLClient";

    private static final int BATTERY_MIN_TO_TRAIN  = 20;
    private static final int BATTERY_FULL_EPOCHS   = 5;
    private static final int BATTERY_LOW_EPOCHS    = 2;
    private static final int BATTERY_MID_THRESHOLD = 50;
    private static final int MAX_REGISTER_RETRIES  = 3;
    private static final int RETRY_DELAY_MS        = 4_000;

    private final Context         context;
    private final String          serverUrl;
    private final String          deviceId;
    private final ExecutorService executor;

    private List<Object> globalWeights = new ArrayList<>();
    private int          currentRound  = -1;

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

    public void runFederatedRound(
            final int numSamples,
            final Map<String, Integer> topicCounts,
            final List<String> docTypes,
            final float localLoss,
            final FLCallback callback) {

        executor.execute(() -> {
            try {
                BatteryInfo battery = getBatteryInfo();
                Log.i(TAG, "Battery: " + battery.level + "% charging=" + battery.isCharging);
                if (!battery.isCharging && battery.level < BATTERY_MIN_TO_TRAIN) {
                    if (callback != null) callback.onError("Battery too low to train.");
                    return;
                }

                // Register with retry for Render cold-start
                boolean registered = false;
                for (int attempt = 1; attempt <= MAX_REGISTER_RETRIES; attempt++) {
                    try {
                        registerDevice(battery);
                        registered = true;
                        break;
                    } catch (Exception e) {
                        Log.w(TAG, "Register attempt " + attempt + " failed: " + e.getMessage());
                        if (attempt < MAX_REGISTER_RETRIES) Thread.sleep(RETRY_DELAY_MS);
                    }
                }
                if (!registered) {
                    if (callback != null) callback.onError("Server unreachable after " + MAX_REGISTER_RETRIES + " attempts.");
                    return;
                }

                boolean pulled = pullGlobalModel();
                if (!pulled) {
                    if (callback != null) callback.onError("Failed to pull global model.");
                    return;
                }

                int epochs = chooseEpochs(battery);
                Log.i(TAG, "Local training: " + epochs + " epochs, " + numSamples + " samples");
                List<Object> updatedWeights = localTrain(globalWeights, epochs, numSamples);

                JSONObject result = uploadWeights(
                        updatedWeights, numSamples, topicCounts, docTypes, localLoss, battery);

                String status = result.optString("status", "unknown");
                int    round  = result.optInt("round", currentRound);
                Log.i(TAG, "Upload result: status=" + status + " round=" + round);

                if (callback != null) callback.onSuccess(status, round);

            } catch (Exception e) {
                Log.e(TAG, "FL round failed: " + e.getMessage(), e);
                if (callback != null) callback.onError(e.getMessage());
            }
        });
    }

    private void registerDevice(BatteryInfo battery) throws Exception {
        JSONObject body = new JSONObject();
        body.put("device_id",     deviceId);
        body.put("battery_level", battery.level);
        body.put("is_charging",   battery.isCharging);
        body.put("model_version", "1.0");
        postJson(serverUrl + "/register", body);
        Log.i(TAG, "Registered device: " + deviceId);
    }

    private boolean pullGlobalModel() {
        try {
            String     response = getRequest(serverUrl + "/get_global_model");
            JSONObject json     = new JSONObject(response);
            currentRound  = json.optInt("round", 0);
            JSONArray arr = json.getJSONArray("weights");
            globalWeights = jsonArrayToList(arr);
            Log.i(TAG, "Pulled global model: round=" + currentRound + " layers=" + globalWeights.size());
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Pull failed: " + e.getMessage(), e);
            return false;
        }
    }

    private int chooseEpochs(BatteryInfo battery) {
        if (battery.isCharging)                     return BATTERY_FULL_EPOCHS;
        if (battery.level >= BATTERY_MID_THRESHOLD) return BATTERY_FULL_EPOCHS - 1;
        if (battery.level >= BATTERY_MIN_TO_TRAIN)  return BATTERY_LOW_EPOCHS;
        return 1;
    }

    @SuppressWarnings("unchecked")
    private List<Object> localTrain(List<Object> weights, int epochs, int numSamples) {
        List<Object> updated = deepCopyWeights(weights);
        double lr = 0.001 * epochs;
        for (Object layer : updated) {
            if (layer instanceof List) applyNoise((List<Object>) layer, lr);
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
            List<Object> weights, int numSamples,
            Map<String, Integer> topicCounts, List<String> docTypes,
            float localLoss, BatteryInfo battery) throws JSONException, IOException {

        JSONObject body = new JSONObject();
        body.put("device_id",     deviceId);
        body.put("weights",       listToJsonArray(weights));
        body.put("num_samples",   numSamples);
        body.put("battery_level", battery.level);
        body.put("is_charging",   battery.isCharging);
        body.put("local_loss",    localLoss);
        body.put("round",         currentRound);

        JSONObject topics = new JSONObject();
        for (Map.Entry<String, Integer> e : topicCounts.entrySet())
            topics.put(e.getKey(), e.getValue());
        body.put("topic_counts", topics);

        JSONArray dtArr = new JSONArray();
        for (String dt : docTypes) dtArr.put(dt);
        body.put("doc_types", dtArr);

        String resp = postJson(serverUrl + "/upload_weights", body);
        return new JSONObject(resp);
    }

    private BatteryInfo getBatteryInfo() {
        IntentFilter filter = new IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED);
        android.content.Intent intent = context.registerReceiver(null, filter);
        int level = 100; boolean charging = true;
        if (intent != null) {
            int lvl   = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
            int scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1);
            level = (scale > 0) ? (int)(lvl * 100f / scale) : 100;
            int s = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1);
            charging = s == BatteryManager.BATTERY_STATUS_CHARGING
                    || s == BatteryManager.BATTERY_STATUS_FULL;
        }
        return new BatteryInfo(level, charging);
    }

    static class BatteryInfo {
        final int level; final boolean isCharging;
        BatteryInfo(int l, boolean c) { level = l; isCharging = c; }
    }

    private String getRequest(String urlStr) throws IOException {
        URL url = new URL(urlStr);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("GET");
        conn.setConnectTimeout(20_000);
        conn.setReadTimeout(30_000);
        try {
            int code = conn.getResponseCode();
            if (code < 200 || code >= 300)
                throw new IOException("GET " + urlStr + " → HTTP " + code + " : " + readStream(conn.getErrorStream()));
            return readStream(conn.getInputStream());
        } finally { conn.disconnect(); }
    }

    private String postJson(String urlStr, JSONObject body) throws IOException {
        URL url = new URL(urlStr);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json; utf-8");
        conn.setDoOutput(true);
        conn.setConnectTimeout(20_000);
        conn.setReadTimeout(60_000);
        try (OutputStream os = conn.getOutputStream()) {
            os.write(body.toString().getBytes(StandardCharsets.UTF_8));
        }
        try {
            int code = conn.getResponseCode();
            if (code >= 200 && code < 300) return readStream(conn.getInputStream());
            throw new IOException("POST " + urlStr + " → HTTP " + code + " : " + readStream(conn.getErrorStream()));
        } finally { conn.disconnect(); }
    }

    private String readStream(InputStream is) {
        if (is == null) return "(no response body)";
        try (Scanner sc = new Scanner(is, "UTF-8")) {
            StringBuilder sb = new StringBuilder();
            while (sc.hasNextLine()) sb.append(sc.nextLine());
            return sb.toString();
        }
    }

    @SuppressWarnings("unchecked")
    private List<Object> jsonArrayToList(JSONArray arr) throws JSONException {
        List<Object> list = new ArrayList<>();
        for (int i = 0; i < arr.length(); i++) {
            Object elem = arr.get(i);
            list.add(elem instanceof JSONArray ? jsonArrayToList((JSONArray) elem) : ((Number) elem).floatValue());
        }
        return list;
    }

    @SuppressWarnings("unchecked")
    private JSONArray listToJsonArray(List<Object> list) throws JSONException {
        JSONArray arr = new JSONArray();
        for (Object elem : list)
            arr.put(elem instanceof List ? listToJsonArray((List<Object>) elem) : elem);
        return arr;
    }

    @SuppressWarnings("unchecked")
    private List<Object> deepCopyWeights(List<Object> src) {
        List<Object> copy = new ArrayList<>();
        for (Object elem : src)
            copy.add(elem instanceof List ? deepCopyWeights((List<Object>) elem) : elem);
        return copy;
    }
}