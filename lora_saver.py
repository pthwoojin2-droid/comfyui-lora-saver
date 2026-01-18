import os
import sys
import requests
from tqdm import tqdm
from urllib.parse import urlparse, unquote

DEFAULT_SAVE_DIR = "models/loras"

def get_filename_from_cd(cd):
    """Get filename from content-disposition"""
    if not cd:
        return None
    fname = None
    if 'filename=' in cd:
        parts = cd.split('filename=')
        if len(parts) > 1:
            fname = parts[1].strip().strip('"').strip("'")
            # Handle potential encoding parameters if present (simple split)
            fname = fname.split(';')[0].strip()
    return fname

def download_file(url, save_dir=DEFAULT_SAVE_DIR):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    print(f"Connecting to {url}...")
    try:
        # Use stream=True to handle large files
        response = requests.get(url, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        filename = get_filename_from_cd(response.headers.get('content-disposition'))
        if not filename:
             path = urlparse(response.url).path # Use effective URL after redirects
             filename = unquote(os.path.basename(path))
        
        if not filename or filename == '':
            print("Could not determine filename from URL or headers.")
            filename = "downloaded_lora.safetensors" # Fallback
            print(f"Using fallback filename: {filename}")
            
        # Ensure extension if missing (optional but good for safety)
        if not os.path.splitext(filename)[1]:
            filename += ".safetensors"

        save_path = os.path.join(save_dir, filename)
        
        if os.path.exists(save_path):
             print(f"File already exists: {save_path}. Skipping.")
             return

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                t.update(len(data))
                file.write(data)
        t.close()
                
        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")
            
        print(f"\nSuccessfully downloaded to: {save_path}")
        
    except Exception as e:
        print(f"\nError downloading {url}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lora_saver.py <url> [save_dir]")
        print("Example: python lora_saver.py https://civitai.com/api/download/models/12345")
    else:
        url = sys.argv[1]
        save_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SAVE_DIR
        download_file(url, save_dir)
