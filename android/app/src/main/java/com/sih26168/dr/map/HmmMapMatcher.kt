package com.sih26168.dr.map

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.PI

/**
 * HMM map matcher — AGENTS.md Step 10.3 / PS "Map-Matching Filter".
 * Emission: N(distance to road node, sigma=15m)
 * Transition: exp(-|dHeading| / 30deg) — heading matters, not just distance.
 * Streaming Viterbi over a fixed candidate window; returns the snapped track.
 */
class HmmMapMatcher(private val graph: RoadGraph) {

    companion object {
        const val SIGMA = 15.0                 // m — GPS-quality emission sigma
        const val HEADING_SCALE = Math.PI / 6  // 30 deg
        const val CANDIDATES = 20              // max candidate nodes per epoch
        const val MATCH_RADIUS_M = 50.0
    }

    data class Fix(val lat: Double, val lon: Double, val matchedNode: Int)

    private val prevCandidates = ArrayList<Int>()
    private var viterbi = DoubleArray(0)
    val matchedTrack = ArrayList<Fix>()

    /** Feed the raw (drifting) DR/GNSS fix; returns snapped fix. */
    fun update(lat: Double, lon: Double, headingRad: Double?): Fix? {
        // candidate nodes: nearest K (linear scan per epoch — city graphs fine on bg thread)
        val cands = nearestK(lat, lon, CANDIDATES, MATCH_RADIUS_M)
        if (cands.isEmpty()) return null

        val emission = DoubleArray(cands.size) { i ->
            val n = cands[i]
            val d = RoadGraph.haversine(lat, lon, graph.lat(n), graph.lon(n))
            exp(-0.5 * (d / SIGMA) * (d / SIGMA))
        }

        if (prevCandidates.isEmpty()) {
            viterbi = emission.copyOf()
        } else {
            val newV = DoubleArray(cands.size)
            for (i in cands.indices) {
                var best = -1.0
                for (j in prevCandidates.indices) {
                    val hFrom = graph.heading(prevCandidates[j], cands[i])
                    val dHead = if (headingRad == null) 0.0 else abs(headingDiff(hFrom, headingRad))
                    val trans = exp(-dHead / HEADING_SCALE)
                    val score = viterbi[j] * trans
                    if (score > best) best = score
                }
                newV[i] = best * emission[i]
            }
            viterbi = newV
        }
        prevCandidates.clear(); prevCandidates.addAll(cands)

        var bestIdx = 0
        for (i in viterbi.indices) if (viterbi[i] > viterbi[bestIdx]) bestIdx = i
        val node = cands[bestIdx]
        val fix = Fix(graph.lat(node), graph.lon(node), node)
        matchedTrack.add(fix)
        return fix
    }

    private fun nearestK(lat: Double, lon: Double, k: Int, radiusM: Double): List<Int> {
        // cheap deg-distance filter then haversine for top-k
        val scored = ArrayList<Pair<Int, Double>>(graph.nodeCount)
        for (i in 0 until graph.nodeCount) {
            val dLat = graph.lat(i) - lat
            val dLon = graph.lon(i) - lon
            val approx = dLat * dLat + dLon * dLon
            scored.add(i to approx)
        }
        scored.sortBy { it.second }
        val out = ArrayList<Int>(k)
        for ((idx, _) in scored) {
            if (out.size >= k) break
            if (RoadGraph.haversine(lat, lon, graph.lat(idx), graph.lon(idx)) <= radiusM) out.add(idx)
        }
        return out
    }

    private fun headingDiff(a: Double, b: Double): Double {
        var d = a - b
        while (d > PI) d -= 2 * PI
        while (d < -PI) d += 2 * PI
        return d
    }
}
