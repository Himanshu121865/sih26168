package com.sih26168.dr.map

import android.content.Context
import org.maplibre.android.geometry.LatLngBounds
import org.maplibre.android.offline.OfflineManager
import org.maplibre.android.offline.OfflineRegion
import org.maplibre.android.offline.OfflineRegionError
import org.maplibre.android.offline.OfflineRegionStatus
import org.maplibre.android.style.sources.TileSet

/**
 * OSM offline region download — "Download this area" (PS: offline map database).
 * Uses MapLibre OfflineManager: downloads tiles for the visible bbox into
 * app-private storage; viewable with no network afterwards.
 */
class OfflineRegionManager(context: Context, private val styleUrl: String) {

    interface Listener {
        fun onProgress(percent: Int)
        fun onComplete(regionName: String)
        fun onError(message: String)
    }

    private val offlineManager = OfflineManager.getInstance(context)

    init {
        // raise the default 100-tile cap for small city-scale regions
        offlineManager.setOfflineMapboxTileCountLimit(6000)
    }

    fun downloadArea(name: String, bounds: LatLngBounds, minZoom: Long, maxZoom: Long, listener: Listener) {
        val definition = OfflineRegion.OfflineRegionDefinition(
            styleUrl,
            bounds,
            minZoom.toDouble(),
            maxZoom.toDouble(),
            1.0f, // pixelRatio
        )
        val metadata = "{\"name\":\"$name\"}".toByteArray(Charsets.UTF_8)
        val region = offlineManager.createOfflineRegion(definition, metadata) { region ->
            region.setObserver(object : OfflineRegion.OfflineRegionObserver {
                override fun onStatusChanged(status: OfflineRegionStatus) {
                    if (status.isComplete) {
                        listener.onComplete(name)
                        region.setObserver(null)
                    } else {
                        val required = status.requiredResourceCount
                        val completed = status.completedResourceCount
                        if (required > 0) listener.onProgress((100 * completed / required).toInt())
                    }
                }

                override fun onError(error: OfflineRegionError) {
                    listener.onError("offline download: ${error.message}")
                }

                override fun mapboxTileCountLimitExceeded(limit: Long) {
                    listener.onError("tile limit exceeded ($limit) — shrink the area")
                }
            })
            region.setDownloadState(OfflineRegion.STATE_ACTIVE)
        }
    }

    fun listRegions(callback: (List<OfflineRegion>) -> Unit) {
        offlineManager.listOfflineRegions { regions -> callback(regions?.toList() ?: emptyList()) }
    }

    fun deleteRegion(region: OfflineRegion, onDone: () -> Unit = {}) {
        region.delete { onDone() }
    }
}

// TileSet import kept for future local mbtiles source wiring (pre-bundled finale city).
@Suppress("unused")
private val unusedTileSetRef: Class<TileSet> = TileSet::class.java
