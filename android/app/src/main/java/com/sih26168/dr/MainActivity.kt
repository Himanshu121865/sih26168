package com.sih26168.dr

import android.Manifest
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.sih26168.dr.engine.DrPipeline
import com.sih26168.dr.io.CsvLogger
import com.sih26168.dr.map.OfflineRegionManager
import com.sih26168.dr.map.RoadGraph
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.maplibre.android.MapLibre
import org.maplibre.android.camera.CameraUpdateFactory
import org.maplibre.android.geometry.LatLng
import org.maplibre.android.geometry.LatLngBounds
import org.maplibre.android.maps.MapView
import org.maplibre.android.maps.Style
import org.maplibre.android.style.layers.LineLayer
import org.maplibre.android.style.sources.GeoJsonSource
import org.maplibre.geojson.Feature
import org.maplibre.geojson.FeatureCollection
import org.maplibre.geojson.LineString
import org.maplibre.geojson.Point
import kotlin.math.sqrt

class MainActivity : AppCompatActivity(), SensorEventListener {

    companion object {
        // Detailed OSM vector style, no key, works offline after download.
        const val STYLE_URL = "https://tiles.openfreemap.org/styles/liberty"
        const val PERMISSIONS = 1001
    }

    private lateinit var mapView: MapView
    private lateinit var statusChip: TextView
    private lateinit var sheetSummary: TextView
    private lateinit var sheetDetail: TextView
    private lateinit var pipeline: DrPipeline
    private lateinit var logger: CsvLogger
    private lateinit var offline: OfflineRegionManager

    private var source: GeoJsonSource? = null
    private val track = ArrayList<Point>()
    private var lastSensorTs = 0L
    private var tStart = 0L
    private var distTraveled = 0.0
    private var lastV = 0.0
    private var lastGnssLat: Double? = null
    private var lastGnssLon: Double? = null
    private var mapReady = false
    private var pendingCenter: LatLng? = null
    private var loadingOverlay: android.view.View? = null
    private var firstFixDone = false

    private val locationClient by lazy { LocationServices.getFusedLocationProviderClient(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        MapLibre.getInstance(this)
        setContentView(R.layout.activity_main)

        statusChip = findViewById(R.id.statusChip)
        sheetSummary = findViewById(R.id.sheetSummary)
        sheetDetail = findViewById(R.id.sheetDetail)
        loadingOverlay = findViewById(R.id.loadingOverlay)
        pipeline = DrPipeline(this, useGravity = true)  // raw Android IMU keeps gravity
        logger = CsvLogger(this)
        offline = OfflineRegionManager(this, STYLE_URL)

        mapView = findViewById(R.id.mapView)
        mapView.onCreate(savedInstanceState)
        mapView.getMapAsync { map ->
            map.setStyle(Style.Builder().fromUri(STYLE_URL), object : Style.OnStyleLoaded {
                override fun onStyleLoaded(style: Style) {
                    mapReady = true
                    source = GeoJsonSource("dr-track")
                    style.addSource(source!!)
                    style.addLayer(
                        LineLayer("track-layer", "dr-track")
                            .withProperties(
                                org.maplibre.android.style.layers.PropertyFactory.lineColor("#1A73E8"),
                                org.maplibre.android.style.layers.PropertyFactory.lineWidth(5f),
                            )
                    )
                    pendingCenter?.let {
                        map.animateCamera(CameraUpdateFactory.newLatLngZoom(it, 16.0))
                        pendingCenter = null
                    }
                    // initial text now that map is ready
                    sheetDetail.visibility = android.view.View.VISIBLE
                    sheetDetail.text = getString(R.string.detail_line, 0f, 0f, 0f)
                }
            })
            map.uiSettings.isCompassEnabled = true
            map.uiSettings.isLogoEnabled = false
            map.uiSettings.isAttributionEnabled = true
            // default to Delhi until GNSS fixes — will be overridden by first fix below
            map.cameraPosition = org.maplibre.android.camera.CameraPosition.Builder()
                .target(LatLng(28.6139, 77.2090)).zoom(11.0).build()
            tryHideLoading()
        }

        findViewById<FloatingActionButton>(R.id.btnLocate).setOnClickListener {
            val target = when {
                lastGnssLat != null && lastGnssLon != null -> LatLng(lastGnssLat!!, lastGnssLon!!)
                pipeline.lat != 0.0 || pipeline.lon != 0.0 -> LatLng(pipeline.lat, pipeline.lon)
                else -> null
            }
            if (target != null) {
                if (mapReady) {
                    mapView.getMapAsync { m -> m.animateCamera(CameraUpdateFactory.newLatLngZoom(target, 16.0), 600) }
                } else {
                    pendingCenter = target
                    Toast.makeText(this, "Map loading…", Toast.LENGTH_SHORT).show()
                }
            } else {
                Toast.makeText(this, "No fix yet — move outdoors", Toast.LENGTH_SHORT).show()
            }
        }

        requestPermissions()
        startSensors()
        startLocation()
        // Auto-hide loading after 3s even without GPS (indoors) — map is usable
        lifecycleScope.launch {
            delay(3000)
            if (!firstFixDone) {
                firstFixDone = true
                tryHideLoading()
            }
        }
        // Tap loading to dismiss
        findViewById<android.view.View>(R.id.loadingOverlay)?.setOnClickListener {
            firstFixDone = true
            tryHideLoading()
        }

        // 10Hz engine ticker: fuse, integrate dead-reckoning pose, refresh UI
        lifecycleScope.launch(Dispatchers.Default) {
            while (true) {
                delay(100)
                val dt = 0.1
                lastV = pipeline.onFusionTick(dt, null, null)
                val course = pipeline.alignment.yawGnss
                if (course != null) {
                    val d = pipeline.ekf.forwardSpeed() * dt
                    val latR = Math.toRadians(pipeline.lat)
                    pipeline.lat += Math.toDegrees(d * kotlin.math.cos(course) / 6371000.0)
                    pipeline.lon += Math.toDegrees(d * kotlin.math.sin(course) /
                        (6371000.0 * kotlin.math.cos(latR)))
                }
                updateUi(pipeline.lat, pipeline.lon)
            }
        }

        lifecycleScope.launch(Dispatchers.IO) {
            try { pipeline.setRoadGraph(RoadGraph.load(this@MainActivity)) } catch (_: Exception) { /* no bundled graph yet */ }
        }
    }

    private fun requestPermissions() {
        val needed = listOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION,
        ).filter { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }
        if (needed.isNotEmpty()) {
            if (needed.any { ActivityCompat.shouldShowRequestPermissionRationale(this, it) }) {
                Toast.makeText(this, "Location needed for navigation + map", Toast.LENGTH_LONG).show()
            }
            ActivityCompat.requestPermissions(this, needed.toTypedArray(), PERMISSIONS)
        } else {
            statusChip.text = getString(R.string.mode_gnss)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSIONS) {
            val granted = grantResults.isNotEmpty() && grantResults.all { it == PackageManager.PERMISSION_GRANTED }
            if (granted) {
                Toast.makeText(this, "Location granted — starting GNSS", Toast.LENGTH_SHORT).show()
                startLocation()
                // center once we have a fix
                lastGnssLat?.let { lat -> lastGnssLon?.let { lon ->
                    mapView.getMapAsync { m -> m.animateCamera(CameraUpdateFactory.newLatLngZoom(LatLng(lat, lon), 16.0)) }
                }}
            } else {
                Toast.makeText(this, "Location denied — map will be offline only", Toast.LENGTH_LONG).show()
                statusChip.text = "No location perm"
            }
        }
    }

    private fun startSensors() {
        val sm = getSystemService(SENSOR_SERVICE) as SensorManager
        val rateUs = (1_000_000_000 / 100)  // 100Hz
        sm.registerListener(this, sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), rateUs)
        sm.registerListener(this, sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE), rateUs)
    }

    private val lastAcc = DoubleArray(3)
    private val lastGyro = DoubleArray(3)

    override fun onSensorChanged(e: SensorEvent) {
        when (e.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAcc[0] = e.values[0].toDouble(); lastAcc[1] = e.values[1].toDouble(); lastAcc[2] = e.values[2].toDouble()
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGyro[0] = e.values[0].toDouble(); lastGyro[1] = e.values[1].toDouble(); lastGyro[2] = e.values[2].toDouble()
            }
            else -> return
        }
        val dt = if (lastSensorTs == 0L) 0.01 else (e.timestamp - lastSensorTs) / 1e9
        lastSensorTs = e.timestamp
        pipeline.onImu(lastAcc, lastGyro, dt)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun tryHideLoading() {
        if (firstFixDone && mapReady) {
            loadingOverlay?.animate()?.alpha(0f)?.setDuration(300)?.withEndAction {
                loadingOverlay?.visibility = android.view.View.GONE
            }?.start()
        }
    }

    private fun startLocation() {
        // Try last known location first — instant center while waiting for fresh fix
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            locationClient.lastLocation.addOnSuccessListener { loc ->
                if (loc != null) {
                    lastGnssLat = loc.latitude; lastGnssLon = loc.longitude
                    if (pipeline.lat == 0.0 && pipeline.lon == 0.0) {
                        pipeline.lat = loc.latitude; pipeline.lon = loc.longitude
                    }
                    val target = LatLng(loc.latitude, loc.longitude)
                    if (mapReady) {
                        mapView.getMapAsync { m -> m.animateCamera(CameraUpdateFactory.newLatLngZoom(target, 15.0)) }
                    } else {
                        pendingCenter = target
                    }
                    firstFixDone = true
                    tryHideLoading()
                }
            }
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) return
        val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000).build()
        try {
            locationClient.requestLocationUpdates(req, object : LocationCallback() {
                override fun onLocationResult(result: LocationResult) {
                    val loc = result.lastLocation ?: return
                    lastGnssLat = loc.latitude; lastGnssLon = loc.longitude
                    pipeline.seamless.onFix(System.nanoTime() / 1_000_000)
                    if (pipeline.lat == 0.0 && pipeline.lon == 0.0) {
                        pipeline.lat = loc.latitude; pipeline.lon = loc.longitude
                    }
                    // Always recenter on first few fixes until loading is gone
                    if (!firstFixDone) {
                        firstFixDone = true
                        tryHideLoading()
                        if (mapReady) {
                            mapView.getMapAsync { m -> m.animateCamera(CameraUpdateFactory.newLatLngZoom(LatLng(loc.latitude, loc.longitude), 16.0)) }
                        } else {
                            pendingCenter = LatLng(loc.latitude, loc.longitude)
                        }
                    }
                    loc.bearing.toDouble().let { pipeline.alignment.updateGnssHeading(Math.toRadians(it), loc.speed.toDouble()) }
                    updateUi(loc.latitude, loc.longitude)
                }
            }, mainLooper)
        } catch (e: SecurityException) {
            Toast.makeText(this, "Location permission missing", Toast.LENGTH_SHORT).show()
        }
    }

    private fun updateUi(gnssLat: Double, gnssLon: Double) {
        // Filter invalid 0,0 (Null Island near Nigeria) from initial pipeline 0,0 before first fix
        val isValidFix = !(gnssLat == 0.0 && gnssLon == 0.0) && gnssLat.isFinite() && gnssLon.isFinite() && kotlin.math.abs(gnssLat) > 0.1
        val v = lastV
        val mode = pipeline.mode
        if (isValidFix) distTraveled += v * 0.1
        runOnUiThread {
            statusChip.text = getString(
                when (mode) {
                    com.sih26168.dr.engine.SeamlessHandler.Mode.GNSS -> R.string.mode_gnss
                    com.sih26168.dr.engine.SeamlessHandler.Mode.INS -> R.string.mode_ins
                }
            )
            statusChip.setTextColor(
                when (mode) {
                    com.sih26168.dr.engine.SeamlessHandler.Mode.GNSS -> 0xFF1A73E8.toInt()
                    com.sih26168.dr.engine.SeamlessHandler.Mode.INS -> 0xFFE8710A.toInt()
                }
            )
            val pos = pipeline.ekf.position()
            val rawSigma = sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2])
            val sigma = if (rawSigma.isFinite()) rawSigma else 0.0  // EKF can be NaN before first GPS
            sheetSummary.text = getString(R.string.summary_line, v.toFloat(), sigma.toFloat())
            sheetDetail.text = getString(
                R.string.detail_line,
                Math.toDegrees(pipeline.lean.phi).toFloat(),
                pipeline.lean.pBike.toFloat(),
                distTraveled.toFloat(),
            )
            if (isValidFix) {
                // Drop initial 0,0 if it slipped in, and avoid duplicate last point
                if (track.isNotEmpty() && track[0].longitude() == 0.0 && track[0].latitude() == 0.0) track.removeAt(0)
                if (track.isEmpty() || track.last().longitude() != gnssLon || track.last().latitude() != gnssLat) {
                    track.add(Point.fromLngLat(gnssLon, gnssLat))
                    // keep last 500 points to avoid memory bloat
                    if (track.size > 500) track.removeAt(0)
                }
                source?.setGeoJson(FeatureCollection.fromFeatures(listOf(Feature.fromGeometry(LineString.fromLngLats(track)))))
            }
        }
        logger.log(
            (System.currentTimeMillis() - tStart) / 1000.0,
            pipeline.lat, pipeline.lon, gnssLat, gnssLon,
            v, pipeline.avnet.sigmaV.toDouble(), pipeline.lean.phi, pipeline.lean.pBike,
            mode.name,
        )
    }

    private fun downloadVisibleArea() {
        mapView.getMapAsync { map ->
            val b = map.projection.visibleRegion.latLngBounds
            Toast.makeText(this, R.string.downloading, Toast.LENGTH_SHORT).show()
            offline.downloadArea(
                "area_" + System.currentTimeMillis() / 1000, b, 14, 17,
                object : OfflineRegionManager.Listener {
                    override fun onProgress(percent: Int) { runOnUiThread { sheetDetail.text = "$percent%" } }
                    override fun onComplete(regionName: String) {
                        runOnUiThread { Toast.makeText(this@MainActivity, R.string.done, Toast.LENGTH_SHORT).show() }
                    }
                    override fun onError(message: String) {
                        runOnUiThread { Toast.makeText(this@MainActivity, message, Toast.LENGTH_LONG).show() }
                    }
                },
            )
        }
    }

    override fun onStart() { super.onStart(); mapView.onStart() }
    override fun onResume() { super.onResume(); mapView.onResume() }
    override fun onPause() { super.onPause(); mapView.onPause() }
    override fun onStop() { super.onStop(); mapView.onStop() }
    override fun onSaveInstanceState(outState: Bundle) { super.onSaveInstanceState(outState); mapView.onSaveInstanceState(outState) }
    override fun onLowMemory() { super.onLowMemory(); mapView.onLowMemory() }
    override fun onDestroy() { super.onDestroy(); mapView.onDestroy(); pipeline.avnet.close() }
}
