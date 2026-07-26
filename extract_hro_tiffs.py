"""
extract_hro_tiffs.py
======================
Automatically scans the downloaded USGS files (which don't have .zip extensions),
finds the high-resolution GeoTIFF (.tif) imagery inside them, and extracts ONLY
the imagery into a clean folder for use in QGIS.

Usage:
    python extract_hro_tiffs.py
"""
import os
import glob
import zipfile
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./hro_tiles", help="Folder containing downloaded USGS files")
    parser.add_argument("--output_dir", type=str, default="./hro_imagery", help="Folder to save extracted GeoTIFFs")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all files in the input directory
    downloaded_files = [f for f in glob.glob(os.path.join(args.input_dir, "*")) if os.path.isfile(f)]
    
    if not downloaded_files:
        print(f"No files found in {args.input_dir}")
        return

    print(f"Found {len(downloaded_files)} files. Scanning for GeoTIFFs...")
    
    extracted_count = 0

    for file_path in downloaded_files:
        filename = os.path.basename(file_path)
        
        # Check if it's a valid zip file (even without the .zip extension)
        if not zipfile.is_zipfile(file_path):
            continue
            
        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                # Find all .tif or .jp2 files inside the zip
                image_files = [f for f in z.namelist() if f.lower().endswith(('.tif', '.jp2'))]
                
                for img_in_zip in image_files:
                    # We just want the filename, not the nested folder structure inside the zip
                    img_basename = os.path.basename(img_in_zip)
                    out_path = os.path.join(args.output_dir, img_basename)
                    
                    if os.path.exists(out_path):
                        print(f"  Skipping {img_basename} (Already extracted)")
                        continue
                        
                    print(f"Extracting {img_basename}...")
                    
                    # Extract the file directly to the output directory without keeping folder structure
                    with z.open(img_in_zip) as source, open(out_path, "wb") as target:
                        shutil.copyfileobj(source, target)
                        
                    extracted_count += 1
                    
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    print("\nExtraction Complete!")
    if extracted_count > 0:
        print(f"Successfully extracted {extracted_count} GeoTIFFs to: {os.path.abspath(args.output_dir)}")
        print("You can now drag and drop all the files in that folder directly into QGIS!")
    else:
        print("No new GeoTIFFs were extracted.")

if __name__ == "__main__":
    main()
