#!/usr/bin/env bash
# Generates the android/ and ios/ folders (not committed) and applies the
# few changes this app needs. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."

flutter create --project-name d_face_lab --org dev.dfacelab --platforms=android,ios .
rm -f test/widget_test.dart

GRADLE=android/app/build.gradle.kts
[ -f "$GRADLE" ] || GRADLE=android/app/build.gradle
# FFmpegKit needs Android 7.0 (API 24) or newer.
sed -i -E 's/minSdk *= *flutter\.minSdkVersion/minSdk = 24/; s/minSdkVersion +flutter\.minSdkVersion/minSdkVersion 24/' "$GRADLE"
grep -n "minSdk" "$GRADLE"

cp tool/android/proguard-rules.pro android/app/proguard-rules.pro

MANIFEST=android/app/src/main/AndroidManifest.xml
# Release builds need INTERNET explicitly (model download); old Androids need
# storage permission to save into the gallery.
if ! grep -q "android.permission.INTERNET" "$MANIFEST"; then
  sed -i 's#<application#<uses-permission android:name="android.permission.INTERNET"/>\n    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" android:maxSdkVersion="29"/>\n    <application android:requestLegacyExternalStorage="true"#' "$MANIFEST"
fi
sed -i 's#android:label="[^"]*"#android:label="D. Face Lab"#' "$MANIFEST"

# Launcher icons.
for d in mdpi hdpi xhdpi xxhdpi xxxhdpi; do
  if [ -f "tool/android/icons/ic_launcher_$d.png" ]; then
    cp "tool/android/icons/ic_launcher_$d.png" "android/app/src/main/res/mipmap-$d/ic_launcher.png"
  fi
done

if [[ "$(uname)" == "Darwin" ]]; then
  PLIST=ios/Runner/Info.plist
  /usr/libexec/PlistBuddy -c "Add :NSPhotoLibraryUsageDescription string 'Per scegliere foto e video da elaborare.'" "$PLIST" || true
  /usr/libexec/PlistBuddy -c "Add :NSPhotoLibraryAddUsageDescription string 'Per salvare i risultati nella galleria.'" "$PLIST" || true
  sed -i '' "s/^# platform :ios.*/platform :ios, '14.0'/" ios/Podfile 2>/dev/null || true
fi

flutter pub get
