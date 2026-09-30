pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        flatDir { dirs("app/libs") }   // for the SNPE/QNN .aar you'll add manually
    }
}

rootProject.name = "MultiModelPipeline"
include(":app")
