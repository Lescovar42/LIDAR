html = """<!DOCTYPE html>
<html>
<head>
    <title>Ridgecrest Earthquake Observations Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body { margin: 0; padding: 0; font-family: sans-serif; }
        #map { width: 100vw; height: 100vh; }
        .map-title {
            position: absolute;
            top: 10px;
            left: 50px;
            z-index: 1000;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            pointer-events: none;
        }
    </style>
</head>
<body>
    <div class="map-title"><h2>Ridgecrest Earthquake Observations</h2></div>
    <div id="map"></div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <script>
        var map = L.map('map').setView([35.65, -117.6], 10);

        var satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles &copy; Esri',
            maxZoom: 19
        }).addTo(map);

        var osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors'
        });
        
        var baseMaps = {
            "Satellite Imagery": satellite,
            "OpenStreetMap": osm
        };
        L.control.layers(baseMaps).addTo(map);

        // Fetch the GeoJSON data
        fetch('ridgecrest_observations.geojson')
            .then(response => response.json())
            .then(geojsonData => {
                var geojsonLayer = L.geoJSON(geojsonData, {
                    onEachFeature: function (feature, layer) {
                        if (feature.properties) {
                            var popupContent = "<b>" + (feature.properties.name || "Observation") + "</b><br>";
                            if (feature.properties.description) {
                                popupContent += feature.properties.description;
                            }
                            layer.bindPopup(popupContent, { maxWidth: 500, maxHeight: 400 });
                        }
                    },
                    pointToLayer: function (feature, latlng) {
                        return L.circleMarker(latlng, {
                            radius: 6,
                            fillColor: "#ff0000",
                            color: "#ffffff",
                            weight: 1,
                            opacity: 1,
                            fillOpacity: 0.8
                        });
                    }
                }).addTo(map);
                
                map.fitBounds(geojsonLayer.getBounds());
            })
            .catch(err => console.error("Error loading geojson: ", err));
    </script>
</body>
</html>"""

with open(r'd:\LIDAR\map_ridgecrest.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("Updated map_ridgecrest.html to use fetch API")
