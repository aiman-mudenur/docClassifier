package com.yourapp.flclient;

import android.content.Context;
import android.net.Uri;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DocumentProcessor
 * -----------------
 * Converts DOCX, PDF, and spreadsheet files into plain text for NLP.
 *
 * Dependencies to add in build.gradle (app):
 *
 *   // Apache POI for DOCX and XLSX (use the Android-compatible trimmed build)
 *   implementation 'org.apache.poi:poi:5.2.5'
 *   implementation 'org.apache.poi:poi-ooxml:5.2.5'
 *
 *   // PdfBox-Android for PDF
 *   implementation 'com.tom_roush:pdfbox-android:2.0.27.0'
 *
 * Initialize PdfBox once in Application.onCreate():
 *   PDFBoxResourceLoader.init(getApplicationContext());
 */
public class DocumentProcessor {

    private static final String TAG = "DocumentProcessor";

    public enum DocType { PDF, DOCX, XLSX, CSV, UNKNOWN }

    public static class ProcessedDoc {
        public final String text;
        public final DocType type;
        public final int wordCount;

        ProcessedDoc(String text, DocType type) {
            this.text      = text;
            this.type      = type;
            this.wordCount = text.split("\\s+").length;
        }
    }

    private final Context context;

    public DocumentProcessor(Context context) {
        this.context = context.getApplicationContext();
    }

    /**
     * Main entry point. Detects file type from URI mime-type or extension,
     * extracts text, returns a ProcessedDoc.
     */
    public ProcessedDoc process(Uri uri) {
        String mime = context.getContentResolver().getType(uri);
        DocType type = detectType(uri, mime);
        String text;
        try {
            switch (type) {
                case DOCX:  text = extractDocx(uri);  break;
                case PDF:   text = extractPdf(uri);   break;
                case XLSX:  text = extractXlsx(uri);  break;
                case CSV:   text = extractCsv(uri);   break;
                default:    text = extractPlainText(uri); break;
            }
        } catch (Exception e) {
            Log.e(TAG, "Extraction failed for " + type + ": " + e.getMessage(), e);
            text = "";
        }
        Log.i(TAG, "Processed " + type + ": " + text.length() + " chars");
        return new ProcessedDoc(text.trim(), type);
    }

    // ── Type detection ────────────────────────────────────────────────────────

    private DocType detectType(Uri uri, String mime) {
        if (mime != null) {
            if (mime.contains("pdf"))   return DocType.PDF;
            if (mime.contains("word") || mime.contains("docx") || mime.contains("openxmlformats-officedocument.wordprocessingml"))
                return DocType.DOCX;
            if (mime.contains("spreadsheet") || mime.contains("excel") || mime.contains("xlsx"))
                return DocType.XLSX;
            if (mime.contains("csv") || mime.contains("comma-separated"))
                return DocType.CSV;
        }
        String path = uri.getPath();
        if (path != null) {
            path = path.toLowerCase();
            if (path.endsWith(".pdf"))  return DocType.PDF;
            if (path.endsWith(".docx")) return DocType.DOCX;
            if (path.endsWith(".xlsx") || path.endsWith(".xls")) return DocType.XLSX;
            if (path.endsWith(".csv"))  return DocType.CSV;
        }
        return DocType.UNKNOWN;
    }

    // ── Extractors ────────────────────────────────────────────────────────────

    /**
     * DOCX extraction via Apache POI XWPFDocument.
     * Extracts paragraphs and table cell text.
     */
    private String extractDocx(Uri uri) throws Exception {
        // Using reflection to avoid hard compile dependency when POI not yet added.
        // Replace with direct POI calls once dependency is added:
        //
        //   try (InputStream is = context.getContentResolver().openInputStream(uri)) {
        //       XWPFDocument doc = new XWPFDocument(is);
        //       StringBuilder sb = new StringBuilder();
        //       for (XWPFParagraph p : doc.getParagraphs())
        //           sb.append(p.getText()).append("\n");
        //       for (XWPFTable t : doc.getTables())
        //           for (XWPFTableRow r : t.getRows())
        //               for (XWPFTableCell c : r.getTableCells())
        //                   sb.append(c.getText()).append(" ");
        //       return sb.toString();
        //   }

        try (InputStream is = context.getContentResolver().openInputStream(uri)) {
            Class<?> docClass = Class.forName("org.apache.poi.xwpf.usermodel.XWPFDocument");
            Object doc = docClass.getConstructor(InputStream.class).newInstance(is);

            // Paragraphs
            List<?> paragraphs = (List<?>) docClass.getMethod("getParagraphs").invoke(doc);
            StringBuilder sb = new StringBuilder();
            Class<?> paraClass = Class.forName("org.apache.poi.xwpf.usermodel.XWPFParagraph");
            for (Object p : paragraphs)
                sb.append(paraClass.getMethod("getText").invoke(p)).append("\n");

            // Tables
            List<?> tables = (List<?>) docClass.getMethod("getTables").invoke(doc);
            Class<?> tableClass = Class.forName("org.apache.poi.xwpf.usermodel.XWPFTable");
            Class<?> rowClass   = Class.forName("org.apache.poi.xwpf.usermodel.XWPFTableRow");
            Class<?> cellClass  = Class.forName("org.apache.poi.xwpf.usermodel.XWPFTableCell");
            for (Object t : tables) {
                List<?> rows = (List<?>) tableClass.getMethod("getRows").invoke(t);
                for (Object r : rows) {
                    List<?> cells = (List<?>) rowClass.getMethod("getTableCells").invoke(r);
                    for (Object c : cells)
                        sb.append(cellClass.getMethod("getText").invoke(c)).append(" | ");
                    sb.append("\n");
                }
            }
            return sb.toString();
        }
    }

    /**
     * PDF extraction via PdfBox-Android.
     * Remember to call PDFBoxResourceLoader.init(context) in Application.onCreate().
     */
    private String extractPdf(Uri uri) throws Exception {
        // Direct PdfBox usage (add import when dependency is present):
        //
        //   try (InputStream is = context.getContentResolver().openInputStream(uri);
        //        PDDocument doc = PDDocument.load(is)) {
        //       PDFTextStripper stripper = new PDFTextStripper();
        //       return stripper.getText(doc);
        //   }

        try (InputStream is = context.getContentResolver().openInputStream(uri)) {
            Class<?> docClass = Class.forName("com.tom_roush.pdfbox.pdmodel.PDDocument");
            Object doc = docClass.getMethod("load", InputStream.class).invoke(null, is);

            Class<?> stripperClass = Class.forName("com.tom_roush.pdfbox.text.PDFTextStripper");
            Object stripper = stripperClass.newInstance();
            String text = (String) stripperClass.getMethod("getText", docClass).invoke(stripper, doc);

            docClass.getMethod("close").invoke(doc);
            return text;
        }
    }

    /**
     * XLSX extraction via Apache POI.
     * Reads all sheets, all rows, all cells as strings.
     */
    private String extractXlsx(Uri uri) throws Exception {
        // Direct POI usage:
        //
        //   try (InputStream is = context.getContentResolver().openInputStream(uri);
        //        Workbook wb = new XSSFWorkbook(is)) {
        //       StringBuilder sb = new StringBuilder();
        //       DataFormatter fmt = new DataFormatter();
        //       for (int si = 0; si < wb.getNumberOfSheets(); si++) {
        //           Sheet sheet = wb.getSheetAt(si);
        //           sb.append("Sheet: ").append(sheet.getSheetName()).append("\n");
        //           for (Row row : sheet)
        //               for (Cell cell : row)
        //                   sb.append(fmt.formatCellValue(cell)).append("\t");
        //           sb.append("\n");
        //       }
        //       return sb.toString();
        //   }

        try (InputStream is = context.getContentResolver().openInputStream(uri)) {
            Class<?> wbClass  = Class.forName("org.apache.poi.xssf.usermodel.XSSFWorkbook");
            Object wb = wbClass.getConstructor(InputStream.class).newInstance(is);

            Class<?> fmtClass = Class.forName("org.apache.poi.ss.usermodel.DataFormatter");
            Object fmt = fmtClass.newInstance();

            int sheets = (int) wbClass.getMethod("getNumberOfSheets").invoke(wb);
            StringBuilder sb = new StringBuilder();

            for (int si = 0; si < sheets; si++) {
                Object sheet = wbClass.getMethod("getSheetAt", int.class).invoke(wb, si);
                Class<?> sheetClass = sheet.getClass();
                sb.append("Sheet: ")
                  .append(sheetClass.getMethod("getSheetName").invoke(sheet))
                  .append("\n");

                Iterable<?> rows = (Iterable<?>) sheetClass.getMethod("iterator").invoke(sheet);
                for (Object row : rows) {
                    Iterable<?> cells = (Iterable<?>) row.getClass().getMethod("iterator").invoke(row);
                    for (Object cell : cells) {
                        String val = (String) fmtClass
                                .getMethod("formatCellValue", Class.forName("org.apache.poi.ss.usermodel.Cell"))
                                .invoke(fmt, cell);
                        sb.append(val).append("\t");
                    }
                    sb.append("\n");
                }
            }
            wbClass.getMethod("close").invoke(wb);
            return sb.toString();
        }
    }

    /**
     * CSV extraction — no library needed, just split on commas.
     */
    private String extractCsv(Uri uri) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (InputStream is = context.getContentResolver().openInputStream(uri);
             BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = br.readLine()) != null) {
                // Replace commas with spaces for NLP
                sb.append(line.replace(",", " ")).append("\n");
            }
        }
        return sb.toString();
    }

    private String extractPlainText(Uri uri) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (InputStream is = context.getContentResolver().openInputStream(uri);
             BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = br.readLine()) != null) sb.append(line).append("\n");
        }
        return sb.toString();
    }

    // ── NLP Preprocessing ─────────────────────────────────────────────────────

    /**
     * Tokenize and clean extracted text.
     * Returns lowercase word tokens with stopwords removed.
     */
    public static List<String> tokenize(String text) {
        String[] STOPWORDS = {
            "a","an","the","is","it","in","of","to","and","or","for",
            "on","at","by","with","as","be","was","are","were","this",
            "that","from","but","not","have","has","had","he","she","they"
        };
        java.util.Set<String> stopSet = new java.util.HashSet<>(java.util.Arrays.asList(STOPWORDS));

        String cleaned = text.toLowerCase().replaceAll("[^a-z0-9\\s]", " ");
        String[] words = cleaned.split("\\s+");
        List<String> tokens = new ArrayList<>();
        for (String w : words) {
            if (w.length() > 2 && !stopSet.contains(w)) tokens.add(w);
        }
        return tokens;
    }

    /**
     * Simple TF-IDF-style 128-dim feature vector from token list.
     * Maps each token to a bucket (hash % 128) and counts term frequency.
     * Normalises to unit length.
     */
    public static float[] toFeatureVector(List<String> tokens, int dims) {
        float[] vec = new float[dims];
        for (String t : tokens) {
            int bucket = Math.abs(t.hashCode()) % dims;
            vec[bucket] += 1.0f;
        }
        // L2 normalise
        float norm = 0f;
        for (float v : vec) norm += v * v;
        norm = (float) Math.sqrt(norm);
        if (norm > 0) for (int i = 0; i < dims; i++) vec[i] /= norm;
        return vec;
    }

    /**
     * Naive topic assignment based on keyword presence.
     * Replace with TFLite inference for real classification.
     * Returns topic index 0-4.
     */
    public static int classifyTopic(List<String> tokens) {
        Map<Integer, String[]> topicKeywords = new HashMap<>();
        topicKeywords.put(0, new String[]{"finance","revenue","profit","budget","cost","expense"});
        topicKeywords.put(1, new String[]{"health","medical","patient","disease","treatment","clinical"});
        topicKeywords.put(2, new String[]{"technology","software","hardware","data","network","system"});
        topicKeywords.put(3, new String[]{"legal","contract","law","regulation","compliance","policy"});
        topicKeywords.put(4, new String[]{"education","student","learning","course","training","school"});

        int[] scores = new int[5];
        for (String token : tokens) {
            for (Map.Entry<Integer, String[]> entry : topicKeywords.entrySet()) {
                for (String kw : entry.getValue()) {
                    if (token.equals(kw)) scores[entry.getKey()]++;
                }
            }
        }
        int best = 0;
        for (int i = 1; i < 5; i++) if (scores[i] > scores[best]) best = i;
        return best;
    }
}
