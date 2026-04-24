package com.example.hybridfl.utils

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import org.apache.poi.xwpf.extractor.XWPFWordExtractor
import org.apache.poi.xwpf.usermodel.XWPFDocument
import java.io.BufferedReader
import java.io.InputStreamReader

object FileUtil {
    
    fun initPdfBox(context: Context) {
        PDFBoxResourceLoader.init(context)
    }

    fun extractTextFromUri(context: Context, uri: Uri): String {
        val fileName = getFileName(context, uri) ?: return ""
        val extension = fileName.substringAfterLast('.', "")
        
        return try {
            when (extension.lowercase()) {
                "txt" -> readTxt(context, uri)
                "pdf" -> readPdf(context, uri)
                "docx" -> readDocx(context, uri)
                else -> "Unsupported file type: $extension"
            }
        } catch (e: Exception) {
            e.printStackTrace()
            "Error extracting text."
        }
    }

    private fun readTxt(context: Context, uri: Uri): String {
        val builder = java.lang.StringBuilder()
        context.contentResolver.openInputStream(uri)?.use { inputStream ->
            BufferedReader(InputStreamReader(inputStream)).use { reader ->
                var line = reader.readLine()
                while (line != null) {
                    builder.append(line).append("\n")
                    line = reader.readLine()
                }
            }
        }
        return builder.toString()
    }

    private fun readPdf(context: Context, uri: Uri): String {
        var text = ""
        context.contentResolver.openInputStream(uri)?.use { inputStream ->
            val document = PDDocument.load(inputStream)
            val stripper = PDFTextStripper()
            text = stripper.getText(document)
            document.close()
        }
        return text
    }

    private fun readDocx(context: Context, uri: Uri): String {
        var text = ""
        context.contentResolver.openInputStream(uri)?.use { inputStream ->
            val document = XWPFDocument(inputStream)
            val extractor = XWPFWordExtractor(document)
            text = extractor.text
            extractor.close()
        }
        return text
    }

    private fun getFileName(context: Context, uri: Uri): String? {
        var name: String? = null
        val cursor = context.contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val index = it.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index != -1) name = it.getString(index)
            }
        }
        return name
    }
}
