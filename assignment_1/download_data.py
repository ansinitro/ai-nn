import os
import urllib.request
import zipfile

def download_jena_climate():
    url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "raw")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data", "processed"), exist_ok=True)
    
    zip_path = os.path.join(data_dir, "jena_climate_2009_2016.csv.zip")
    csv_path = os.path.join(data_dir, "jena_climate_2009_2016.csv")
    
    if not os.path.exists(csv_path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Cleaning up zip file...")
        os.remove(zip_path)
        print(f"Done! Dataset is located at: {csv_path}")
    else:
        print(f"Dataset already exists at: {csv_path}")

if __name__ == "__main__":
    download_jena_climate()
