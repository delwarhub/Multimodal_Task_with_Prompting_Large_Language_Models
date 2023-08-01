import os
import yaml
import json
import pickle
import subprocess
from typing import Literal

def load_from_yaml(path_to_file: str):
    print(f"loading data from .yaml file @ {path_to_file}")
    with open(path_to_file) as file:
        _dict = yaml.safe_load(file)
    return _dict

def load_from_txt(path_to_file: str):
    print(f"loading data from .txt file @ {path_to_file}")
    with open (path_to_file, "r") as myfile:
        data = myfile.read().splitlines()
    return data

def save_to_json(path_to_file: str, data: list):
    with open(path_to_file, 'w') as outfile:
        json.dump(data, outfile)
    print(f"file saved @ loc: {path_to_file}")

def load_from_json(path_to_file: str):
    print(f"loading data from .json file @ {path_to_file}")
    with open(path_to_file, "r") as json_file:
        _dict = json.load(json_file)
    return _dict

def save_to_pickle(data_list, path_to_file):
    with open(path_to_file, 'wb') as file:
        pickle.dump(data_list, file)
    print(f"file saved @ loc: {path_to_file}")

def load_from_pickle(path_to_file):
    print(f"loading data from .pkl file @ {path_to_file}")
    with open(path_to_file, 'rb') as file:
        data_list = pickle.load(file)
    return data_list

data_types_ = Literal["questions", "annotations", "images"]

def download_and_extract_data(data_type: data_types_="questions", url: str=None):
    type_2_directory = {
        "questions": "./Questions",
        "annotations": "./Annotations",
        "images": "./Images"
    }
    directory_name = type_2_directory[data_type]
    if os.path.exists(directory_name):
        # check if the folder already exists & delete
        cmd_result_0 = subprocess.run(["rm", "-rf", directory_name], text=True)
        assert cmd_result_0.returncode == 0; f"command failed to execute: {cmd_result_0.returncode}"
    # create new directory w/ attributed data_type
    cmd_result_1 = subprocess.run(["mkdir", directory_name], text=True)
    assert cmd_result_1.returncode == 0; f"command failed to execute: {cmd_result_1.returncode}"
    # download specific data under the directory using the attributed url path
    cmd_result_2 = subprocess.run(["wget", "-P", directory_name, url], text=True)
    assert cmd_result_2.returncode == 0; f"command failed to execute: {cmd_result_2.returncode}"
    # extract previously downloaded zip-file corresponding to the attributed data_type
    zip_file_name = os.listdir(directory_name)[0]
    path_to_zip_file = os.path.join(directory_name, zip_file_name)
    cmd_result_3 = subprocess.run(["unzip", path_to_zip_file, "-d", directory_name])
    assert cmd_result_3.returncode == 0; f"command failed to execute: {cmd_result_3.returncode}"
    print(f"{data_type} downloaded and extracted successfully @ loc: {directory_name}")

# _ = [download_and_extract_data(data_type=data_type, url=url) for data_type, url in zip(config["DATA_TYPES"], config["DATA_TYPE_URLS"])]

config = load_from_yaml("./config.yaml")