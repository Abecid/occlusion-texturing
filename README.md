# Occlusion Texturing

## Setup
```
pip install -r requirements.txt

# If git-lts is not installed
sudo apt-get install git-lfs

git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models

# Download Sample Objaverse Objects
python download_objaverse.py
```
- Refer to "./configs/default.yaml" to setup your configuration file

## 1. Get Rendered Images from Plane Intersections
```
python ray_casting.py
```

## 2. Depth to Image Generation 
```
python stylization.py
```

## 3. Mesh Reconstruction via UV Mapping (xAtlas)