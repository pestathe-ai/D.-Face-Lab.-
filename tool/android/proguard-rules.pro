# Native libraries call back into these classes through JNI: keep them intact.
-keep class ai.onnxruntime.** { *; }
-keep class com.masicai.flutteronnxruntime.** { *; }
-keep class com.antonkarpenko.ffmpegkit.** { *; }
-dontwarn ai.onnxruntime.**
-dontwarn com.antonkarpenko.ffmpegkit.**
