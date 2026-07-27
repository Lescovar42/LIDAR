import zipfile
import os

kmz_path = r'd:\LIDAR\KMZ_File_Ridgecrest_Observations_Slip_Prov_Rel_1\Ridgecrest_Observations_Slip_Prov_Rel_1.kmz'
extract_dir = r'd:\LIDAR\scratch'

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

with zipfile.ZipFile(kmz_path, 'r') as kmz:
    kmz.extractall(extract_dir)
    print(f"Extracted to {extract_dir}:")
    print(kmz.namelist())
