package com.sih26168.dr.engine

/**
 * SO(3)/SE2(3) math — Kotlin port of python/inekf_harness.py
 * (which is a validated port of ref/QAIIMU lie_group_utils.py).
 * All matrices are row-major Array<DoubleArray> (3x3).
 */
object LieGroup {

    fun skew(w: DoubleArray): Array<DoubleArray> = arrayOf(
        doubleArrayOf(0.0, -w[2], w[1]),
        doubleArrayOf(w[2], 0.0, -w[0]),
        doubleArrayOf(-w[1], w[0], 0.0),
    )

    fun matMul(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val out = Array(3) { DoubleArray(3) }
        for (i in 0..2) for (j in 0..2) {
            var s = 0.0
            for (k in 0..2) s += a[i][k] * b[k][j]
            out[i][j] = s
        }
        return out
    }

    fun matVec(a: Array<DoubleArray>, v: DoubleArray): DoubleArray {
        val out = DoubleArray(3)
        for (i in 0..2) out[i] = a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2]
        return out
    }

    fun matTranspose(a: Array<DoubleArray>): Array<DoubleArray> {
        val out = Array(3) { DoubleArray(3) }
        for (i in 0..2) for (j in 0..2) out[j][i] = a[i][j]
        return out
    }

    fun matAdd(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        Array(3) { i -> DoubleArray(3) { j -> a[i][j] + b[i][j] } }

    fun matScale(a: Array<DoubleArray>, s: Double): Array<DoubleArray> =
        Array(3) { i -> DoubleArray(3) { j -> a[i][j] * s } }

    fun eye3(): Array<DoubleArray> = arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0),
        doubleArrayOf(0.0, 0.0, 1.0),
    )

    /** exp([phi]_x) via Rodrigues. */
    fun so3exp(phi: DoubleArray): Array<DoubleArray> {
        val ang = norm(phi)
        val K = skew(phi)
        if (ang < 1e-10) {
            return matAdd(eye3(), K)
        }
        val axis = doubleArrayOf(phi[0] / ang, phi[1] / ang, phi[2] / ang)
        val Kk = skew(axis)
        val s = kotlin.math.sin(ang)
        val c = kotlin.math.cos(ang)
        // R = c*I + (1-c)*axis*axis^T + s*K
        val out = Array(3) { i -> DoubleArray(3) { j ->
            c * (if (i == j) 1.0 else 0.0) + (1 - c) * axis[i] * axis[j] + s * Kk[i][j]
        } }
        return out
    }

    fun norm(v: DoubleArray): Double = kotlin.math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    fun cross(a: DoubleArray, b: DoubleArray): DoubleArray = doubleArrayOf(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )

    /**
     * SE2(3) exp for 9D xi = [phi(3), rho_v(3), rho_p(3)] -> (R, v, p).
     * Exact left Jacobian (matches python sen3exp).
     */
    fun sen3exp(xi: DoubleArray): Triple<Array<DoubleArray>, DoubleArray, DoubleArray> {
        val phi = doubleArrayOf(xi[0], xi[1], xi[2])
        val ang = norm(phi)
        val K = skew(phi)
        val J: Array<DoubleArray>
        val R: Array<DoubleArray>
        if (ang < 1e-10) {
            val K2 = matMul(K, K)
            J = matAdd(matAdd(eye3(), matScale(K, 0.5)), matScale(K2, 1.0 / 6.0))
            R = matAdd(matAdd(eye3(), K), matScale(K2, 0.5))
        } else {
            val s = kotlin.math.sin(ang)
            val c = kotlin.math.cos(ang)
            val ang2 = ang * ang
            val axis = doubleArrayOf(phi[0] / ang, phi[1] / ang, phi[2] / ang)
            val outer = Array(3) { i -> DoubleArray(3) { j -> axis[i] * axis[j] } }
            J = Array(3) { i -> DoubleArray(3) { j ->
                (s / ang) * (if (i == j) 1.0 else 0.0) +
                    (1 - s / ang) * outer[i][j] + ((1 - c) / ang) * K[i][j]
            } }
            R = Array(3) { i -> DoubleArray(3) { j ->
                c * (if (i == j) 1.0 else 0.0) + (1 - c) * outer[i][j] + s * K[i][j]
            } }
        }
        val rhoV = doubleArrayOf(xi[3], xi[4], xi[5])
        val rhoP = doubleArrayOf(xi[6], xi[7], xi[8])
        return Triple(R, matVec(J, rhoV), matVec(J, rhoP))
    }
}
