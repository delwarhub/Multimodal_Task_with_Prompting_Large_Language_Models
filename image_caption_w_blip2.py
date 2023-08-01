import os
import re
import pandas as pd
import torch
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from utils import load_from_json

def extract_response_yes_no(input_string: str):
    # synonyms for 'yes' and 'no'
    synonyms_yes = ['yes', 'yeah', 'yep', 'affirmative', 'positive']
    synonyms_no = ['no', 'nope', 'negative']
    # regular expression pattern to search for 'yes' or 'no' synonyms
    pattern = r'\b(?:' + '|'.join(synonyms_yes + synonyms_no) + r')\b'
    match = re.search(pattern, input_string, re.IGNORECASE)
    if match:
        response = match.group(0).lower()
        if response in synonyms_yes:
            return "yes"
        elif response in synonyms_no:
            return "no"
    return "no" # returning no-humans as output for human-presence instead of None

def extract_response_outside_inside(input_string: str):
    # synonyms for 'outside' and 'inside'
    synonyms_outside = ['outside', 'outdoors', 'exterior', 'beyond']
    synonyms_inside = ['inside', 'indoors', 'interior', 'within']
    # regular expression pattern to search for 'outside' or 'inside' synonyms
    pattern = r'\b(?:' + '|'.join(synonyms_outside + synonyms_inside) + r')\b'
    match = re.search(pattern, input_string, re.IGNORECASE)
    if match:
        response = match.group(0).lower()
        if response in synonyms_outside:
            return "outside"
        elif response in synonyms_inside:
            return "inside"
    return "outside" # returning outside as output for locations instead of None

def extract_option(input_string: str):
    # regex pattern to match the desired substring in the format (a) or (b)
    pattern = r'\([a-zA-Z]\)'
    match = re.search(pattern, input_string)
    return match.group() if match else None

class Image_Caption_w_Blip2:
    def __init__(self, config):
        super(Image_Caption_w_Blip2, self).__init__()
        
        self.config = config
        self.data = pd.read_csv(self.config["PATH_TO_SAMPLED_PREPROCESSED_DATA"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # intialize Blip2 model and visual processors
        self.model_type = self.config["blip2_model_type"].split("_")[-1]
        self.load_model()
        # load cached prompt for generating captions
        self.caption_prompts = load_from_json("./data/caption_prompts.json")
        self.attribute_list = list(self.caption_prompts.keys())
        # generate caption data
        self.output_caption_data = self.batch_processing()
        
    def load_model(self):
        self.model, self.vis_processors, _ = load_model_and_preprocess(name=self.config["blip2_architecture"],
                                                                       model_type=self.config["blip2_model_type"],
                                                                       is_eval=True, device=self.device)

    def formulate_image_filename(self, image_id: str):
        max_length = 12
        padded_image_id = str(image_id).zfill(max_length)
        output = "COCO_val2014_" + padded_image_id + ".jpg"
        return output.strip()

    def load_and_process_image(self, image_id: str):
        path_to_image = os.path.join(self.config["PATH_TO_VALIDATION_IMAGES"], self.formulate_image_filename(image_id))
        raw_image = Image.open(path_to_image).convert("RGB")
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        return raw_image, image

    def post_process_blip2_captions(self, generated_captions: list):
        processed_captions = []
        skip_indices = []
        for index, (attribute, caption_value) in enumerate(zip(self.attribute_list, generated_captions)):
            if attribute == "color_category":
                retrieved_option = extract_option(caption_value)
                option_mapping = {
                    "(a)": "Color Image",
                    "(b)": "Black & White Image",
                }
                option_category = option_mapping.get(retrieved_option, "Color Image")
                processed_captions.append(option_category)
            elif attribute == "human_presence":
                human_presence_response = extract_response_yes_no(caption_value)
                if human_presence_response == "no":
                    skip_indices.extend([10, 11, 12, 14]) # skip human related attribute checks/captions (age will also be excluded!)
                processed_captions.append(human_presence_response)
            elif attribute == "locations":
                location_category = extract_response_outside_inside(caption_value)
                if location_category == "outside":
                    skip_indices.extend([8]) # skip indoor lighting conditions
                elif location_category == "inside":
                    skip_indices.extend([7]) # skip outdoor weather
                processed_captions.append(location_category)
            else:
                processed_captions.append(caption_value)
        # processed_captions = [caption_info for index, caption_info in enumerate(processed_captions) if index not in skip_indices]
        # caption_attribute_list = [attribute for index, attribute in enumerate(caption_attribute_list) if index not in skip_indices]
        return processed_captions, skip_indices

    def generate_captions(self, image: torch.Tensor):
        # process image and appropriate prompt through BLIP2 
        path_1, path_2, path_3 = [], [], []
        for _, prompt in self.caption_prompts.items():
            output = self.model.generate({"image": image, "prompt": prompt}, max_length=100, temperature=0.5, top_p=0.5, num_captions=3, num_beams=5, repetition_penalty=2.0)
            path_1.append(output[0])
            path_2.append(output[1])
            path_3.append(output[2])
        # post processing prompt and decision making
        path_1 = self.post_process_blip2_captions(path_1)
        path_2 = self.post_process_blip2_captions(path_2)
        path_3 = self.post_process_blip2_captions(path_3)
        return path_1, path_2, path_3

    def batch_processing(self):

        def convert_path_2_dict(caption_path_list):
            caption_path_dict = defaultdict(list)
            skip_index_list = []
            for item in tqdm(caption_path_list):
                data, skip_indices = item
                skip_index_list.append(skip_indices)
                for attribute, text in zip(self.attribute_list, data):
                    caption_path_dict[attribute].append(text)
            return pd.DataFrame(caption_path_dict), skip_index_list
        
        output_1, output_2, output_3 = [], [], []
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            _, image = self.load_and_process_image(str(row["image_id"]))
            path_1, path_2, path_3 = self.generate_captions(image)
            output_1.append(path_1)
            output_2.append(path_2)
            output_3.append(path_3)
            
        # append data into a pd.DataFrame and cache the skip-index list's
        output_caption_data = dict()
        for index, item in enumerate([output_1, output_2, output_3]):
            output_caption_data[f"output_{index+1}"] = convert_path_2_dict(caption_path_list=item)
        return output_caption_data