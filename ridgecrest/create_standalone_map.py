import json
import base64

with open(r'd:\LIDAR\ridgecrest_observations.geojson', 'r', encoding='utf-8') as f:
    geojson_data = json.load(f)

json_str = json.dumps(geojson_data)
b64_data = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')

html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ridgecrest Earthquake Observations Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body {{ margin: 0; padding: 0; font-family: sans-serif; }}
        #map {{ width: 100vw; height: 100vh; }}
        .map-title {{
            position: absolute;
            top: 10px;
            left: 50px;
            z-index: 1000;
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 15px rgba(0,0,0,0.2);
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <div class="map-title"><h2>Ridgecrest Earthquake Observations</h2></div>
    <div id="map"></div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

    <script>
        // Safely decode the base64 GeoJSON data
        var b64Data = "{b64_data}";
        // Decode base64 to string, avoiding UTF-8 issues by using a proper decoder if needed,
        // but for standard ASCII/UTF-8 JSON, this simple decoding works:
        var jsonString = decodeURIComponent(escape(window.atob(b64Data)));
        var geojsonData = JSON.parse(jsonString);

        var map = L.map('map').setView([35.65, -117.6], 10);

        var satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles &copy; Esri',
            maxZoom: 19
        }}).addTo(map);

        var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors'
        }});
        
        var baseMaps = {{
            "Satellite Imagery": satellite,
            "OpenStreetMap": osm
        }};
        L.control.layers(baseMaps).addTo(map);

        var geojsonLayer = L.geoJSON(geojsonData, {{
            onEachFeature: function (feature, layer) {{
                if (feature.properties) {{
                    var popupContent = "<b>" + (feature.properties.name || "Observation") + "</b><br>";
                    if (feature.properties.description) {{
                        popupContent += feature.properties.description;
                    }}
                    layer.bindPopup(popupContent, {{ maxWidth: 500, maxHeight: 400 }});
                }}
            }},
            pointToLayer: function (feature, latlng) {{
                return L.circleMarker(latlng, {{
                    radius: 6,
                    fillColor: "#ff0000",
                    color: "#ffffff",
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                }});
            }}
        }}).addTo(map);
        
        map.fitBounds(geojsonLayer.getBounds());
    </script>
</body>
</html>"""

with open(r'd:\LIDAR\Standalone_Ridgecrest_Map.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("Created Standalone_Ridgecrest_Map.html")
