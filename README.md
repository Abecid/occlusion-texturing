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
```
- Refer to "./configs/default.yaml" to setup your configuraion file
### (Optional)
```
# Download pytorch3d using the compatible python, cuda, pytorch versions with your system
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/download.html
```

## 1. Get Rendered Images from Plane Intersections
```
python ray_casting.py
```

## 2. Depth to Image Generation 
```
python stylization.py
```

## 3. Mesh Reconstruction via UV Mapping (xAtlas)