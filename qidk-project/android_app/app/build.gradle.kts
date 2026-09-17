plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.multipipeline"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.multipipeline"
        minSdk = 31          // QIDK SM8650 ships Android 14 (API 34); minSdk 31 is safe/plenty
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        ndk {
            // SM8650 is arm64 only - no point shipping other ABIs
            abiFilters += listOf("arm64-v8a")
        }
    }

    // Force physical extraction of native libraries to the filesystem so Qualcomm FastRPC can load HTP Skel/Stub binaries
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    // Prevent AAPT from compressing the large .dlc neural network assets
    androidResources {
        noCompress += listOf("dlc")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    // The SNPE/QNN native .so files must land in jniLibs/arm64-v8a - see android_app/README.md
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // CameraX - live camera feed
    val cameraxVersion = "1.3.4"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")

    // SNPE / QNN Java API - NOT on Maven. Copy snpe-release.aar from
    // $QNN_SDK_ROOT/lib/android into app/libs/ before building (see README.md).
    implementation(files("libs/snpe-release.aar"))
}
