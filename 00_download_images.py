import pandas as pd
import requests
import os
from urllib.parse import urlparse
from pathlib import Path
from uuid_extensions import uuid7, uuid7str
from tqdm.auto import tqdm
from os.path import splitext

tqdm.pandas()

def download_image(url, folder):
    if pd.isna(url):
        return None
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Extract filename from URL
            # filename = os.path.basename(urlparse(url).path)
            # if not filename:
            #     filename = 'image.jpg'  # Default filename if not found in URL
            name, ext = os.path.splitext(url)
            if '?' in ext:
                ext = ext.split("?")[0]
            filename = uuid7str()+ext

            # Ensure the filename is unique
            file_path = Path(folder) / filename
            counter = 1
            while file_path.exists():
                file_path = Path(folder) / f"{name}_{counter}{ext}"
                counter += 1
            
            # Save the image
            with open(file_path, 'wb') as file:
                file.write(response.content)
            
            return str(file_path)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    
    return None

# Load the dataframe
data = pd.read_csv('claim_matching_dataset.csv')  # Replace with your actual file path

# Create images folder if it doesn't exist
image_folder = 'cr_images'
os.makedirs(image_folder, exist_ok=True)

# Download images and create new column
data['local_image_path'] = data['cr_image'].apply(lambda x: download_image(x, image_folder))

# Save the updated dataframe
data.to_csv('updated_data.csv', index=False)

print("Image download complete. Updated dataframe saved to 'updated_data.csv'.")