package com.sih26168.dr.engine

/**
 * 21-DOF right-invariant EKF — Kotlin port of python/inekf_harness.py InEKF
 * (validated 2026-08-30, commit f8a18d9; original: QAIIMU filter_*_improved).
 *
 * State layout: [R_nav 0:3 | v 3:6 | p 6:9 | b_g 9:12 | b_a 12:15 | R_car 15:18 | p_car 18:21]
 *
 * CRITICAL (see AGENTS.md build notes):
 * - measurement z = [v_fwd, v_lat, 0] — our frame has X=forward (NOT ref's Y-forward)
 * - G depends on data source: raw Android IMU keeps gravity -> G = -9.8 (useGravity=true).
 * - bias clamping: b_g ±0.5 rad/s, b_a ±2.0 m/s^2
 */
class InEKFEngine(useGravity: Boolean = true) {

    companion object {
        const val DIM = 21
        const val NOISE_DIM = 18
        const val BG_CLAMP = 0.5
        const val BA_CLAMP = 2.0
        val G = doubleArrayOf(0.0, 0.0, -9.80665)
    }

    // rotation 3x3 row-major
    var R: Array<DoubleArray> = LieGroup.eye3(); private set
    var v = DoubleArray(3); private set
    var p = DoubleArray(3); private set
    var bg = DoubleArray(3); private set
    var ba = DoubleArray(3); private set
    private var Rcar: Array<DoubleArray> = LieGroup.eye3()
    private var pcar = DoubleArray(3)

    private val P = Array(DIM) { DoubleArray(DIM) }
    private val g = if (useGravity) G.copyOf() else DoubleArray(3)

    init {
        // P0 diag: att 1e-2, vel 0.5, pos 0.5, bg 1e-4, ba 1e-2, R_car 1e-6, p_car 1e-6
        val p0 = doubleArrayOf(
            1e-2, 1e-2, 1e-2,
            0.5, 0.5, 0.5,
            0.5, 0.5, 0.5,
            1e-4, 1e-4, 1e-4,
            1e-2, 1e-2, 1e-2,
            1e-6, 1e-6, 1e-6,
            1e-6, 1e-6, 1e-6,
        )
        for (i in 0 until DIM) P[i][i] = p0[i]
    }

    private val qAcc: Double = 0.5  // was 30.0 — caused P explosion to 3e26 when stationary (now fixed)
    private val qNoise = doubleArrayOf(
        1e-4, 1e-4, 1e-4,           // gyro ARW
        qAcc, qAcc, qAcc,           // accel
        1e-6, 1e-6, 1e-6,           // bg walk
        1e-4, 1e-4, 1e-4,           // ba walk
        1e-8, 1e-8, 1e-8,           // R_car
        1e-8, 1e-8, 1e-8,           // p_car
    )

    fun setInitialVelocity(vfwd: Double) {
        v = doubleArrayOf(vfwd, 0.0, 0.0)
    }

    /** Port of filter_propagate_improved. gyro/acc = raw body-frame measurement. */
    fun propagate(gyro: DoubleArray, acc: DoubleArray, dt: Double) {
        val w = doubleArrayOf(gyro[0] - bg[0], gyro[1] - bg[1], gyro[2] - bg[2])
        val dR = LieGroup.so3exp(doubleArrayOf(w[0] * dt, w[1] * dt, w[2] * dt))
        val Rprop = LieGroup.matMul(R, dR)
        val a = doubleArrayOf(acc[0] - ba[0], acc[1] - ba[1], acc[2] - ba[2])
        val aBody = LieGroup.matVec(R, a)
        val aNav = doubleArrayOf(aBody[0] + g[0], aBody[1] + g[1], aBody[2] + g[2])
        val vprop = doubleArrayOf(v[0] + aNav[0] * dt, v[1] + aNav[1] * dt, v[2] + aNav[2] * dt)
        val pprop = doubleArrayOf(
            p[0] + (v[0] + vprop[0]) * dt * 0.5,
            p[1] + (v[1] + vprop[1]) * dt * 0.5,
            p[2] + (v[2] + vprop[2]) * dt * 0.5,
        )

        // F Jacobian (21x21, only non-zero blocks built into flat ops)
        val F = Array(DIM) { DoubleArray(DIM) }
        val skewG = LieGroup.skew(g)
        for (i in 0..2) {
            for (j in 0..2) {
                F[3 + i][0 + j] = skewG[i][j]
                F[6 + i][3 + j] = if (i == j) 1.0 else 0.0
                F[0 + i][9 + j] = -R[i][j]
                F[3 + i][12 + j] = R[i][j]
            }
            val svR = LieGroup.matMul(LieGroup.skew(v), R)
            val spR = LieGroup.matMul(LieGroup.skew(p), R)
            for (j in 0..2) {
                F[3 + i][9 + j] = svR[i][j]
                F[6 + i][9 + j] = spR[i][j]
            }
        }
        for (i in 0 until DIM) for (j in 0 until DIM) F[i][j] *= dt

        // Phi = I + F + F^2/2 + F^3/6
        val phi = phiMatrix(F)

        // Gn (21x18)
        val Gn = Array(DIM) { DoubleArray(NOISE_DIM) }
        for (i in 0..2) {
            for (j in 0..2) {
                Gn[0 + i][0 + j] = -R[i][j]
                Gn[3 + i][3 + j] = R[i][j]
                Gn[9 + i][6 + j] = if (i == j) 1.0 else 0.0
                Gn[12 + i][9 + j] = if (i == j) 1.0 else 0.0
                Gn[15 + i][12 + j] = Rcar[i][j]
                Gn[18 + i][15 + j] = if (i == j) 1.0 else 0.0
            }
        }
        for (i in 0 until DIM) for (j in 0 until NOISE_DIM) Gn[i][j] *= dt

        // P = Phi (P + Gn Q Gn^T) Phi^T   (Q diag folded: GnQ = Gn with columns scaled by qNoise)
        val GnQ = scale(Gn, qNoise)
        val inner = add(P, matTmulMat(Gn, GnQ))   // P + Gn*Q*Gn^T
        P.copyFrom(mul(mul(phi, inner), transpose(phi)))

        R = Rprop; v = vprop; p = pprop
        clampBiases()
    }

    /**
     * Port of filter_update_improved with our frame convention.
     * z = [v_fwd, v_lat, 0]; rDiag = matching 3-variance diag.
     */
    fun updateVelocity(z: DoubleArray, rDiag: DoubleArray) {
        val Rt = LieGroup.matTranspose(R)
        val vImu = LieGroup.matVec(Rt, v)             // body-frame velocity
        val vCarPred = LieGroup.matVec(Rcar, vImu)    // p_car = 0

        // H: 3x21, H[:, 3:6] = R_car * R_nav^T
        val H = Array(3) { DoubleArray(DIM) }
        val Hn = LieGroup.matMul(Rcar, Rt)
        for (i in 0..2) for (j in 0..2) H[i][3 + j] = Hn[i][j]

        // S = H P H^T + R ; K = P H^T S^-1
        val HP = mulRows(H, P)                 // 3x21
        val S = Array(3) { i ->
            DoubleArray(3) { j ->
                var s = 0.0
                for (k in 0 until DIM) s += HP[i][k] * H[j][k]
                s + (if (i == j) rDiag[i] else 0.0)
            }
        }
        val PHt = Array(DIM) { i ->
            DoubleArray(3) { j ->
                var s = 0.0
                for (k in 0 until DIM) s += P[i][k] * H[j][k]
                s
            }
        }
        val K = solve3(S, PHt)  // DIM x 3

        val innov = doubleArrayOf(
            z[0] - vCarPred[0], z[1] - vCarPred[1], z[2] - vCarPred[2],
        )
        val dx = DoubleArray(DIM)
        for (i in 0 until DIM) {
            var s = 0.0
            for (k in 0..2) s += K[i][k] * innov[k]
            dx[i] = s
        }

        val result = LieGroup.sen3exp(dx.copyOfRange(0, 9))
        val dR = result.first
        val dv = result.second
        val dp = result.third
        R = LieGroup.matMul(dR, R)
        val vRot = LieGroup.matVec(dR, v)
        v = doubleArrayOf(vRot[0] + dv[0], vRot[1] + dv[1], vRot[2] + dv[2])
        val pRot = LieGroup.matVec(dR, p)
        p = doubleArrayOf(pRot[0] + dp[0], pRot[1] + dp[1], pRot[2] + dp[2])
        for (i in 0..2) {
            bg[i] += dx[9 + i]
            ba[i] += dx[12 + i]
        }
        Rcar = LieGroup.matMul(LieGroup.so3exp(dx.copyOfRange(15, 18)), Rcar)
        for (i in 0..2) pcar[i] += dx[18 + i]

        // P = (I-KH) P (I-KH)^T + K R K^T
        val IKH = Array(DIM) { i -> DoubleArray(DIM) { j -> (if (i == j) 1.0 else 0.0) - kh(i, j, K, H) } }
        val A = mul(IKH, P)
        val AT = transpose(A)
        val Pnew = mul(A, AT)
        for (i in 0 until DIM) for (j in 0 until DIM) {
            var s = 0.0
            for (k in 0..2) s += K[i][k] * rDiag[k] * K[j][k]
            P[i][j] = Pnew[i][j] + s
        }
        // symmetrize
        for (i in 0 until DIM) for (j in 0 until DIM) {
            val avg = (P[i][j] + P[j][i]) * 0.5
            P[i][j] = avg; P[j][i] = avg
        }
        clampBiases()
    }

    /** Forward speed in car frame — used for along-track distance. */
    fun forwardSpeed(): Double = LieGroup.matVec(LieGroup.matTranspose(R), v)[0]

    fun position(): DoubleArray = p.copyOf()

    private fun kh(i: Int, j: Int, K: Array<DoubleArray>, H: Array<DoubleArray>): Double {
        var s = 0.0
        for (k in 0..2) s += K[i][k] * H[k][j]
        return s
    }

    private fun phiMatrix(F: Array<DoubleArray>): Array<DoubleArray> {
        val F2 = mul(F, F)
        val F3 = mul(F2, F)
        val out = Array(DIM) { i -> DoubleArray(DIM) { j ->
            val eye = if (i == j) 1.0 else 0.0
            eye + F[i][j] + 0.5 * F2[i][j] + F3[i][j] / 6.0
        } }
        return out
    }

    private fun mul(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val n = a.size; val m = b[0].size; val k2 = b.size
        val out = Array(n) { DoubleArray(m) }
        for (i in 0 until n) for (j in 0 until m) {
            var s = 0.0
            for (k in 0 until k2) s += a[i][k] * b[k][j]
            out[i][j] = s
        }
        return out
    }

    private fun mulRows(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> = mul(a, b)

    private fun transpose(a: Array<DoubleArray>): Array<DoubleArray> {
        val n = a.size; val m = a[0].size
        val out = Array(m) { DoubleArray(n) }
        for (i in 0 until n) for (j in 0 until m) out[j][i] = a[i][j]
        return out
    }

    private fun add(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        Array(a.size) { i -> DoubleArray(a[0].size) { j -> a[i][j] + b[i][j] } }

    private fun scale(a: Array<DoubleArray>, diag: DoubleArray): Array<DoubleArray> =
        Array(a.size) { i -> DoubleArray(a[0].size) { j -> a[i][j] * diag[j] } }

    /** out = a * b^T where b pre-multiplied by diag q (i.e., Gn*Q*Gn^T). */
    private fun matTmulMat(a: Array<DoubleArray>, bPre: Array<DoubleArray>): Array<DoubleArray> {
        // bPre = Gn*Q (Q diag folded by caller); we need Gn * Q * Gn^T = bPre * Gn^T
        val n = a.size; val m = a[0].size
        val out = Array(n) { DoubleArray(n) }
        for (i in 0 until n) for (j in 0 until n) {
            var s = 0.0
            for (k in 0 until m) s += bPre[i][k] * a[j][k]
            out[i][j] = s
        }
        return out
    }

    /** Solve S x = B for x (S is 3x3 symmetric PD, B is DIMx3 -> returns DIMx3). */
    private fun solve3(S: Array<DoubleArray>, B: Array<DoubleArray>): Array<DoubleArray> {
        // Gauss-Jordan with partial pivot on 3x3, applied to each RHS column.
        val aug = Array(3) { DoubleArray(3 + DIM) }
        for (i in 0..2) {
            for (j in 0..2) aug[i][j] = S[i][j]
            for (r in 0 until DIM) aug[i][3 + r] = B[r][i]
        }
        for (col in 0..2) {
            var piv = col
            for (r in col + 1..2) if (kotlin.math.abs(aug[r][col]) > kotlin.math.abs(aug[piv][col])) piv = r
            if (piv != col) { val t = aug[col]; aug[col] = aug[piv]; aug[piv] = t }
            val d = aug[col][col]
            if (kotlin.math.abs(d) < 1e-12) continue
            for (j in col until 3 + DIM) aug[col][j] /= d
            for (r in 0..2) {
                if (r == col) continue
                val f = aug[r][col]
                if (f != 0.0) for (j in col until 3 + DIM) aug[r][j] -= f * aug[col][j]
            }
        }
        val out = Array(DIM) { DoubleArray(3) }
        for (r in 0 until DIM) for (c in 0..2) out[r][c] = aug[c][3 + r]
        return out
    }

    private fun clampBiases() {
        for (i in 0..2) {
            bg[i] = bg[i].coerceIn(-BG_CLAMP, BG_CLAMP)
            ba[i] = ba[i].coerceIn(-BA_CLAMP, BA_CLAMP)
        }
    }
}

private fun Array<DoubleArray>.copyFrom(other: Array<DoubleArray>) {
    for (i in indices) System.arraycopy(other[i], 0, this[i], 0, other[i].size)
}
