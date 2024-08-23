import os
import glob
import gzip
import json
import multiprocessing
import os
import urllib.request
import warnings
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import trimesh


BASE_PATH = os.path.join("./assets/objaverse")
os.makedirs(BASE_PATH, exist_ok=True)

__version__ = "<REPLACE_WITH_VERSION>"
_VERSIONED_PATH = os.path.join(BASE_PATH, "hf-objaverse-v1")
os.makedirs(_VERSIONED_PATH, exist_ok=True)



def glb2obj(glb_path, obj_path):
    mesh = trimesh.load(glb_path)
    
    if isinstance(mesh, trimesh.Scene):
        vertices = 0
        for g in mesh.geometry.values():
            vertices += g.vertices.shape[0]
    elif isinstance(mesh, trimesh.Trimesh):
        vertices = mesh.vertices.shape[0]
    else:
        raise ValueError(f'{glb_path} is not mesh or scene')
    
    # if vertices > 100000:
    #     print(f'Too many vertices in {glb_path}. Skip this mesh')
    #     del mesh, vertices
    #     return 0
    if not os.path.exists(os.path.dirname(obj_path)):
        os.makedirs(os.path.dirname(obj_path))
    mesh.export(obj_path)
    
    del mesh, vertices
    return 1



def load_annotations(uids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load the full metadata of all objects in the dataset.

    Args:
        uids: A list of uids with which to load metadata. If None, it loads
        the metadata for all uids.
    """
    metadata_path = os.path.join(_VERSIONED_PATH, "metadata")
    object_paths = _load_object_paths()
    dir_ids = (
        set([object_paths[uid].split("/")[1] for uid in uids])
        if uids is not None
        else [f"{i // 1000:03d}-{i % 1000:03d}" for i in range(160)]
    )
    if len(dir_ids) > 10:
        dir_ids = tqdm(dir_ids)
    out = {}
    for i_id in dir_ids:
        json_file = f"{i_id}.json.gz"
        local_path = os.path.join(metadata_path, json_file)
        if not os.path.exists(local_path):
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{i_id}.json.gz"
            # wget the file and put it in local_path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(hf_url, local_path)
        with gzip.open(local_path, "rb") as f:
            data = json.load(f)
        if uids is not None:
            data = {uid: data[uid] for uid in uids if uid in data}
        out.update(data)
        if uids is not None and len(out) == len(uids):
            break
    return out


def _load_object_paths() -> Dict[str, str]:
    """Load the object paths from the dataset.

    The object paths specify the location of where the object is located
    in the Hugging Face repo.

    Returns:
        A dictionary mapping the uid to the object path.
    """
    object_paths_file = "object-paths.json.gz"
    local_path = os.path.join(_VERSIONED_PATH, object_paths_file)
    if not os.path.exists(local_path):
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
        # wget the file and put it in local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        object_paths = json.load(f)
    return object_paths


def load_uids() -> List[str]:
    """Load the uids from the dataset.

    Returns:
        A list of uids.
    """
    return list(_load_object_paths().keys())


def _download_object(
    uid: str,
    object_path: str,
    total_downloads: float,
    start_file_count: int,
) -> Tuple[str, str]:
    """Download the object for the given uid.

    Args:
        uid: The uid of the object to load.
        object_path: The path to the object in the Hugging Face repo.

    Returns:
        The local path of where the object was downloaded.
    """
    # print(f"downloading {uid}")
    local_path = os.path.join(_VERSIONED_PATH, object_path)
    tmp_local_path = os.path.join(_VERSIONED_PATH, object_path + ".tmp")
    hf_url = (
        f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_path}"
    )
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(hf_url, tmp_local_path)

    os.rename(tmp_local_path, local_path)

    files = glob.glob(os.path.join(_VERSIONED_PATH, "glbs", "*", "*.glb"))
    print(
        "Downloaded",
        len(files) - start_file_count,
        "/",
        total_downloads,
        "objects",
    )

    return uid, local_path


def load_objects(uids: List[str], download_processes: int = 1) -> Dict[str, str]:
    """Return the path to the object files for the given uids.

    If the object is not already downloaded, it will be downloaded.

    Args:
        uids: A list of uids.
        download_processes: The number of processes to use to download the objects.

    Returns:
        A dictionary mapping the object uid to the local path of where the object
        downloaded.
    """
    object_paths = _load_object_paths()
    out = {}
    if download_processes == 1:
        uids_to_download = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn("Could not find object with uid. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(_VERSIONED_PATH, object_path)
            if os.path.exists(local_path):
                out[uid] = local_path
                continue
            uids_to_download.append((uid, object_path))
        if len(uids_to_download) == 0:
            return out
        start_file_count = len(
            glob.glob(os.path.join(_VERSIONED_PATH, "glbs", "*", "*.glb"))
        )
        for uid, object_path in uids_to_download:
            uid, local_path = _download_object(
                uid, object_path, len(uids_to_download), start_file_count
            )
            out[uid] = local_path
    else:
        args = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn("Could not find object with uid. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(_VERSIONED_PATH, object_path)
            if not os.path.exists(local_path):
                args.append((uid, object_paths[uid]))
            else:
                out[uid] = local_path
        if len(args) == 0:
            return out
        print(
            f"starting download of {len(args)} objects with {download_processes} processes"
        )
        start_file_count = len(
            glob.glob(os.path.join(_VERSIONED_PATH, "glbs", "*", "*.glb"))
        )
        args = [(*arg, len(args), start_file_count) for arg in args]
        with multiprocessing.Pool(download_processes) as pool:
            r = pool.starmap(_download_object, args)
            for uid, local_path in r:
                out[uid] = local_path
    return out


def load_lvis_annotations() -> Dict[str, List[str]]:
    """Load the LVIS annotations.

    If the annotations are not already downloaded, they will be downloaded.

    Returns:
        A dictionary mapping the LVIS category to the list of uids in that category.
    """
    hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/lvis-annotations.json.gz"
    local_path = os.path.join(_VERSIONED_PATH, "lvis-annotations.json.gz")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        lvis_annotations = json.load(f)
    return lvis_annotations



if __name__ == '__main__':    
    # key: uid, value: mesh_name
    uid_mesh_name_dict = {
        "e4f348e98ceb45e3abc77da5b738f1b2": "glasses",
        "8ec8f8a2022d491fa0d74279c4f07304": "human_model1",
        "83801b3d03004e4a8e5617534cd2f068": "airpot",
        "1d3ad2a23c444d5abfabf1a48eeb8c84": "wood_mug",
        "7830358a83ac427cbdd515ddb027ca52": "human_model2",
        "0aecee43ac2749499a16ab4388a0baa2": "temple",
        "12a4585f80cc4f38b6a2b6bdd5cfd368": "headphone",
        "77b6a9cb27f6433bbf48c6ed8fc47827": "skull",
        "d595b87c10d8405c96920dccbae45b17": "mixtape",
        "f2573d07dc5b437b9115ab4251e8f5f8": "t-shirts",
        "de9588f97f644bd689fc6502f262575d": "spike_shoe",
        "6cdd3d87ff9e4b8c8dd80a7c66e6c303": "city_hall",
        "966522c1de144d43a5c5a9d68b7eff5f": "vans_shoe",
        "ffedf6c2e1de40bcaef5c57301ca94fd": "space_rocket",
        "192f7e81dce84c6780434f692a0f96c5": "SUV",
        "0723b35415b0462eb5c01140b6b70340": "old_chair",
        "616761063ad845fe84e1d965bdce5595": "pickup_truck",
        "9871bf23b8a04d30a0a9b9731bae1d14": "Akatsuki_coat",
        "d8c69ea106464c49aedb06e9164e017c": "luggage",
        "be48f3f906ed431b98b1bf03ab7aadd6": "biplane",
        "efa778ca442945b09300a70e34e68532": "Dior_handbag",
        "ed6a25f396a84d808977aa6d074d610e": "Torino1999_bus",
        "3e7b686e96634117a529c1223a8f85fb": "minivan",
        "27120b66ac4e4385889f5542df0c2038": "black_robe",
        "72b40b95e0574fab92efc350196ea2fc": "game_house",
        "f2d3018a83f3422394a36edb6be8a834": "armsofa",
        "7c91bd719e594fcc930f3646db843265": "giraffe",
        "7a0f2d413b5846f591b40580f783c53c": "tree_stump",
        "ea66e95a48aa43aeb0cbd8b4a2664b29": "gundam",
        "faef9fe5ace445e7b2989d1c1ece361c": "shiba_dog",
        "4ec412fc7e0648e0a7a4554143185c33": "goblet",
    }

    obj_dict = load_objects(uid_mesh_name_dict.keys(), download_processes=10)
    
    mesh_path_list = glob.glob(os.path.join(_VERSIONED_PATH, "glbs", "*", "*.glb"))
    convert_success = 0
    for mesh_path in tqdm(mesh_path_list, desc="Converting glb to obj"):
        uid = mesh_path.split("/")[-1].split(".")[0]
        mesh_name = uid_mesh_name_dict[uid]
        os.makedirs(f"{BASE_PATH}/{mesh_name}", exist_ok=True)
        os.system(f"mv {mesh_path} {BASE_PATH}/{mesh_name}/model.glb")
        print(mesh_name)
        convert_success += glb2obj(f"{BASE_PATH}/{mesh_name}/model.glb", f"{BASE_PATH}/{mesh_name}/model.obj")
    
    os.system(f"rm -r {_VERSIONED_PATH}")
    print(f"{len(obj_dict)} meshes downloaded at {BASE_PATH}")
    print(f"{convert_success}/{len(obj_dict)} meshes converted to obj")
