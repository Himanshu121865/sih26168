plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.sih26168.dr"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.sih26168.dr"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
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
tasks.register<Copy>("copyModelAssets") {
    from(rootProject.file("../model.tflite"))
    from(rootProject.file("../scaler.json"))
    if (rootProject.file("../python/hmm/road_graph.json").exists()) {
        from(rootProject.file("../python/hmm/road_graph.json")) { into "maps" }
    }
    into("src/main/assets")
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
}
