import xml.etree.ElementTree as ET
import json
import os

kml_file = r'd:\LIDAR\scratch\Ridgecrest_Observations_Slip_Prov_Rel_1.kml'
geojson_file = r'd:\LIDAR\ridgecrest_observations.geojson'

tree = ET.parse(kml_file)
root = tree.getroot()
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

features = []
for pm in root.findall('.//kml:Placemark', ns):
    name_el = pm.find('kml:name', ns)
    desc_el = pm.find('kml:description', ns)
    pt_el = pm.find('kml:Point/kml:coordinates', ns)
    
    name = name_el.text if name_el is not None else ''
    desc = desc_el.text if desc_el is not None else ''
    
    if pt_el is not None:
        coords_str = pt_el.text.strip()
        parts = coords_str.split(',')
        if len(parts) >= 2:
            lon = float(parts[0])
            lat = float(parts[1])
            
            features.append({
                "type": "Feature",
                "properties": {
                    "name": name,
                    "description": desc
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                }
            })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open(geojson_file, 'w', encoding='utf-8') as f:
    json.dump(geojson, f)

print(f"Exported {len(features)} features to {geojson_file}")
