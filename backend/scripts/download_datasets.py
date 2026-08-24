import os
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'datasets')
os.makedirs(DATA_DIR, exist_ok=True)

DATASETS = {
    "AI4I_2020": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
        "filename": "ai4i2020.csv",
        "is_zip": False
    },
    "NASA_CMAPSS": {
        "url": "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+.zip",
        "filename": "cmapss.zip",
        "is_zip": True
    },
    "Hydraulic_Systems": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00447/data.zip",
        "filename": "hydraulic.zip",
        "is_zip": True
    }
}

def download_datasets():
    for name, info in DATASETS.items():
        print(f"--- Processing {name} ---")
        dataset_path = os.path.join(DATA_DIR, name)
        os.makedirs(dataset_path, exist_ok=True)
        
        file_path = os.path.join(dataset_path, info["filename"])
        
        if not os.path.exists(file_path):
            print(f"Downloading {name} from {info['url']}...")
            try:
                urllib.request.urlretrieve(info["url"], file_path)
                print(f"Downloaded to {file_path}")
            except Exception as e:
                print(f"Failed to download {name}: {e}")
                continue
        else:
            print(f"{name} already downloaded at {file_path}")
            
        if info["is_zip"]:
            extracted_marker = os.path.join(dataset_path, ".extracted")
            if not os.path.exists(extracted_marker):
                print(f"Extracting {name}...")
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(dataset_path)
                    # Mark as extracted
                    with open(extracted_marker, 'w') as f:
                        f.write("done")
                    print(f"Extracted {name}")
                except Exception as e:
                    print(f"Failed to extract {name}: {e}")
            else:
                print(f"{name} already extracted.")

if __name__ == "__main__":
    print(f"Downloading official datasets to: {DATA_DIR}")
    download_datasets()
    print("All datasets processed.")
