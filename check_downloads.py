"""
check_downloads.py
===================
Checks the status of previously requested USGS EarthExplorer downloads
and downloads them if they are ready!

Usage:
    python check_downloads.py
"""
import requests
import os
import getpass
import argparse
import concurrent.futures

M2M_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"
LABEL = "m2m_hro_download"

def send_request(endpoint, payload, api_key=None):
    url = f"{M2M_URL}/{endpoint}"
    headers = {"X-Auth-Token": api_key} if api_key else {}
    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("errorCode"):
        raise Exception(f"M2M API Error ({data['errorCode']}): {data['errorMessage']}")
    return data.get("data")

def download_file(args):
    url, out_path, index, total = args
    if os.path.exists(out_path):
        print(f"[{index}/{total}] File already exists locally, skipping {os.path.basename(out_path)}...")
        return
        
    print(f"[{index}/{total}] Starting {os.path.basename(out_path)}...")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        last_print_mb = 0
        
        # 1MB chunks for gigabit speeds instead of 8KB
        chunk_size = 1048576 
        
        with open(out_path, 'wb') as file:
            for data in response.iter_content(chunk_size=chunk_size):
                if data:
                    size = file.write(data)
                    downloaded += size
                    downloaded_mb = downloaded / (1024 * 1024)
                    if downloaded_mb - last_print_mb > 50: # Print every 50MB to avoid console spam
                        total_mb_str = f" / {total_size / (1024 * 1024):.1f} MB" if total_size else ""
                        print(f"[{index}/{total}] ... {downloaded_mb:.1f} MB{total_mb_str}")
                        last_print_mb = downloaded_mb
                        
        print(f"[{index}/{total}] Finished {os.path.basename(out_path)}!")
    except Exception as e:
        print(f"[{index}/{total}] Failed: {str(e)}")
        # Delete partial file so it can be retried later
        if os.path.exists(out_path):
            os.remove(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", type=str, default="./hro_tiles", help="Output directory")
    args = parser.parse_args()

    token = os.environ.get("USGS_TOKEN")
    if not token:
        token = input("Enter EROS Application Token (it will be visible as you type): ")
    
    username = os.environ.get("USGS_USERNAME")
    if not username:
        username = input("Enter EROS Username: ")

    token = token.strip()
    username = username.strip()

    login_payload = {"username": username, "token": token}

    print("Logging in to USGS M2M API...")
    try:
        api_key = send_request("login-token", login_payload)
        print("Login successful.")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    try:
        print(f"\nChecking status of downloads requested with label '{LABEL}'...")
        retrieve_payload = {"label": LABEL}
        
        results = send_request("download-retrieve", retrieve_payload, api_key)
        
        available = results.get("available", [])
        requested = results.get("requested", [])
        
        print(f"Status: {len(available)} files READY, {len(requested)} files STILL PROCESSING.")
        
        if len(requested) > 0:
            print("USGS is still pulling some files from their archives. You can check back later.")
            
        if not available:
            print("\nNo files are ready to download yet.")
            return
            
        print(f"\n{len(available)} files are ready to download!")
        os.makedirs(args.download_dir, exist_ok=True)
        
        # Prepare arguments for parallel download
        download_args = []
        for i, dl in enumerate(available, 1):
            url = dl.get("url")
            filename = url.split("/")[-1] if "/" in url else f"download_{i}.zip"
            filename = filename.split("?")[0]
            out_path = os.path.join(args.download_dir, filename)
            download_args.append((url, out_path, i, len(available)))
            
        print(f"Downloading up to 8 files in parallel to max out bandwidth...\n")
        
        # Download in parallel using 8 threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(download_file, download_args)
            
        print("\nAll available downloads completed!")
                
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        
    finally:
        print("\nLogging out...")
        send_request("logout", {}, api_key)
        print("Logged out successfully.")

if __name__ == "__main__":
    main()
