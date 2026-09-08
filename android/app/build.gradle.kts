plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.sih26168.dr"
    compileSdk = 34

    // Single version truth: git tag (CI passes APP_VERSION=v* on release).
    // Local builds read "0.1.0-dev"; pyproject version is bumped at tag time.
    val appVersion: String = System.getenv("APP_VERSION") ?: "0.1.0-dev"

    defaultConfig {
        applicationId = "com.sih26168.dr"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = appVersion
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            // Signed release when a keystore is configured (env or local.properties);
            // falls back to debug signing for local field-test builds so adb install -r
            // upgrades in place. CI has no keystore secrets yet (see release.yml TODO).
            val ksFile = System.getenv("ANDROID_STORE_FILE")
                ?: rootProject.file("sih26168-release.jks").takeIf { it.exists() }?.absolutePath
            if (ksFile != null) {
                signingConfig = signingConfigs.create("release") {
                    storeFile = file(ksFile)
                    storePassword = System.getenv("ANDROID_STORE_PASSWORD") ?: "sih26168field2026"
                    keyAlias = System.getenv("ANDROID_KEY_ALIAS") ?: "sih26168"
                    keyPassword = System.getenv("ANDROID_KEY_PASSWORD") ?: "sih26168field2026"
                }
            } else {
                signingConfig = signingConfigs.getByName("debug")
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
}

// Copy the trained model + scaler from the repo root into assets before build.
// (Keeps a single source of truth: python/export_tflite.py output.)
// P2: FAIL the build when scaler.json lacks the current spec fingerprint —
// a stale pair must never reach an APK. (python/core/spec.py is the truth.)
tasks.register<Copy>("copyModelAssets") {
    from(rootProject.file("../model.tflite"))
    from(rootProject.file("../scaler.json"))
    rootProject.file("../model_manifest.json").takeIf { it.exists() }?.let { from(it) }
    if (rootProject.file("../python/hmm/road_graph.json").exists()) {
        from(rootProject.file("../python/hmm/road_graph.json")) { into("maps") }
    }
    into("src/main/assets")
    doFirst {
        val scaler = rootProject.file("../scaler.json")
        if (!scaler.exists()) {
            throw GradleException("scaler.json missing at repo root — run preprocess.py first")
        }
        val text = scaler.readText()
        val expectedVer = 2
        // STRICT_SPEC=1 turns the warning into the hard P2 refusal. Hard-gate
        // by default once the spec-v2 retrain lands (docs/WINDOW_SPEC.md rollout).
        val strict = (System.getenv("STRICT_SPEC") ?: "0") == "1"
        val ver = Regex("\"spec_version\"\\s*:\\s*(\\d+)").find(text)?.groupValues?.get(1)?.toIntOrNull()
        when {
            ver == null -> {
                val msg = "scaler.json has NO spec fingerprint — regenerate with preprocess.py (spec v$expectedVer)"
                if (strict) throw GradleException(msg) else logger.lifecycle("WARN [spec]: $msg")
            }
            ver < expectedVer -> {
                val msg = "scaler.json spec_version=$ver < $expectedVer — stale scaler; retrain + re-export (P2)"
                if (strict) throw GradleException(msg) else logger.lifecycle("WARN [spec]: $msg")
            }
        }
    }
}
tasks.named("preBuild") { dependsOn("copyModelAssets") }

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // TFLite — AVNetLite inference
    implementation("org.tensorflow:tensorflow-lite:2.14.0")

    // MapLibre GL — OSM map rendering + offline regions
    implementation("org.maplibre.gl:android-sdk:11.8.1")
    implementation("org.maplibre.gl:android-sdk-geojson:3.0.1")

    // Location — FusedLocationProvider (GNSS)
    implementation("com.google.android.gms:play-services-location:21.3.0")

    testImplementation("junit:junit:4.13.2")
    // Real org.json for JVM unit tests (android.jar stubs throw "not mocked").
    testImplementation("org.json:json:20240303")
}
