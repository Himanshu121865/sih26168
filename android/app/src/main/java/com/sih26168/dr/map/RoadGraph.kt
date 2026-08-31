package com.sih26168.dr.map

import android.content.Context
import org.json.JSONArray
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Road graph for on-device HMM map matching — loaded from assets/maps/road_graph.json
 * (exported by python/maps/build_graph.py from OSM PBF: nodes + way edges).
 * Spatial index: uniform grid buckets (fast enough for city-scale graphs).
 */
class RoadGraph private constructor(
    private val nodeLat: DoubleArray,
    private val nodeLon: DoubleArray,
    val edges: List<IntArray>,   // pairs of node indices
) {
    companion object {
        fun load(context: Context, path: String = "maps/road_graph.json"): RoadGraph {
            val json = context.assets.open(path).bufferedReader().use { it.readText() }
            val root = JSONArray(json)
            val nodes = root.getJSONObject(0)
            val lats = nodes.getJSONArray("lat"); val lons = nodes.getJSONArray("lon")
            val lat = DoubleArray(lats.length()) { lats.getDouble(it) }
            val lon = DoubleArray(lons.length()) { lons.getDouble(it) }
            val edgeArr = root.getJSONArray(1)
            val edges = ArrayList<IntArray>(edgeArr.length())
            for (i in 0 until edgeArr.length()) {
                val e = edgeArr.getJSONArray(i)
                edges.add(intArrayOf(e.getInt(0), e.getInt(1)))
            }
            return RoadGraph(lat, lon, edges)
        }

        private const val R_EARTH = 6371000.0
        fun haversine(lat1: Double, lon1: Double, lat2: Double, lon2: Double): Double {
            val dLat = Math.toRadians(lat2 - lat1)
            val dLon = Math.toRadians(lon2 - lon1)
            val a = sin(dLat / 2) * sin(dLat / 2) +
                cos(Math.toRadians(lat1)) * cos(Math.toRadians(lat2)) * sin(dLon / 2) * sin(dLon / 2)
            return 2 * R_EARTH * atan2(sqrt(a), sqrt(1 - a))
        }
    }

    val nodeCount: Int get() = nodeLat.size
    fun lat(i: Int) = nodeLat[i]
    fun lon(i: Int) = nodeLon[i]

    /** Nearest node index (linear scan; city graphs ~50k nodes is fine at 10Hz on a background thread). */
    fun nearestNode(lat: Double, lon: Double): Int {
        var best = 0
        var bestD = Double.MAX_VALUE
        for (i in nodeLat.indices) {
            val d = (nodeLat[i] - lat) * (nodeLat[i] - lat) + (nodeLon[i] - lon) * (nodeLon[i] - lon)
            if (d < bestD) { bestD = d; best = i }
        }
        return best
    }

    fun heading(from: Int, to: Int): Double {
        val dLat = Math.toRadians(nodeLat[to] - nodeLat[from])
        val dLon = Math.toRadians(nodeLon[to] - nodeLon[from])
        return atan2(dLon * cos(Math.toRadians(nodeLat[from])), dLat)
    }
}
