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
        // Free OSM raster style (no API key). Swap for a local/offline style at the finale.
        const val STYLE_URL = "https://demotiles.maplibre.org/style.json"
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

    private val locationClient by lazy { LocationServices.getFusedLocationProviderClient(this) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        MapLibre.getInstance(this)
        setContentView(R.layout.activity_main)

        statusChip = findViewById(R.id.statusChip)
        sheetSummary = findViewById(R.id.sheetSummary)
        sheetDetail = findViewById(R.id.sheetDetail)
        pipeline = DrPipeline(this, useGravity = true)  // raw Android IMU keeps gravity
        logger = CsvLogger(this)
        offline = OfflineRegionManager(this, STYLE_URL)

        mapView = findViewById(R.id.mapView)
        mapView.onCreate(savedInstanceState)
        mapView.getMapAsync { map ->
            map.setStyle(Style.Builder().fromUri(STYLE_URL)) { style ->
                source = GeoJsonSource("dr-track")
                style.addSource(source!!)
                style.addLayer(
                    LineLayer("track-layer", "dr-track")
                        .withProperties(
                            org.maplibre.android.style.layers.PropertyFactory.lineColor("#1A73E8"),
                            org.maplibre.android.style.layers.PropertyFactory.lineWidth(5f),
                        )
                )
            }
            map.uiSettings.isCompassEnabled = true
        }

        findViewById<FloatingActionButton>(R.id.btnLocate).setOnClickListener {
            mapView.getMapAsync { map ->
                map.animateCamera(CameraUpdateFactory.newLatLngZoom(LatLng(pipeline.lat, pipeline.lon), 17.0))
            }
        }
        findViewById<FloatingActionButton>(R.id.btnDownload).setOnClickListener { downloadVisibleArea() }

        requestPermissions()
        startSensors()
        startLocation()

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
        val needed = listOf(Manifest.permission.ACCESS_FINE_LOCATION).filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needed.isNotEmpty()) ActivityCompat.requestPermissions(this, needed.toTypedArray(), PERMISSIONS)
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

    private fun startLocation() {
        val req = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000).build()
        locationClient.requestLocationUpdates(req, object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val loc = result.lastLocation ?: return
                pipeline.seamless.onFix(System.nanoTime() / 1_000_000)
                if (pipeline.lat == 0.0 && pipeline.lon == 0.0) {
                    pipeline.lat = loc.latitude; pipeline.lon = loc.longitude
                }
                loc.bearing.toDouble().let { pipeline.alignment.updateGnssHeading(Math.toRadians(it), loc.speed.toDouble()) }
                updateUi(loc.latitude, loc.longitude)
            }
        }, mainLooper)
    }

    private fun updateUi(gnssLat: Double, gnssLon: Double) {
        val v = lastV
        distTraveled += v * 0.1
        runOnUiThread {
            val mode = pipeline.mode
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
            val sigma = sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2])  // |p| = DR drift from start
            sheetSummary.text = getString(R.string.summary_line, v.toFloat(), sigma.toFloat())
            sheetDetail.text = getString(
                R.string.detail_line,
                Math.toDegrees(pipeline.lean.phi).toFloat(),
                pipeline.lean.pBike.toFloat(),
                distTraveled.toFloat(),
            )
            track.add(Point.fromLngLat(gnssLon, gnssLat))
            source?.setGeoJson(FeatureCollection.from(listOf(Feature.fromGeometry(LineString.fromLngLats(track)))))
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
