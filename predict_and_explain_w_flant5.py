import os
from typing import Literal
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from image_caption_w_blip2 import Image_Caption_w_Blip2

flan_model_types_ = Literal["xl", "xxl"]
answer_columns_ = Literal["multiple_choice_answer", "prediction"]


class Predict_and_Explain_w_FlanT5:
    def __init__(self, config):
        super(Predict_and_Explain_w_FlanT5, self).__init__()
        
        self.config = config
        self.data = pd.read_csv(self.config["PATH_TO_SAMPLED_PREPROCESSED_DATA"])
        # load flant5-type model and tokenizer
        print("loading flant5xxl model & tokenizer")
        self.load_flan_t5(model_type="xxl")
        # load & prepare the captions
        print("loading Blip2 and generating captions")
        image_caption_w_blip2 = Image_Caption_w_Blip2(config)
        # predict answers for {caption, question} pair
        print("generating answer for (question, caption) pairs")
        predictions = self.batch_processing_prediction(caption_data=image_caption_w_blip2.output_caption_data["output_3"], data=self.data)
        self.data["prediction"] = predictions
        # perform explanation generation for {caption, question, answer} triplet
        # perform explanation generation for ground-truth multipl-choice-answer selection
        for index, key in enumerate(["output_1", "output_2", "output_3"]):
            print(f"explanation generation for (question, ground-truth-answer, caption) triplet: {index}")
            explanations = self.batch_processing_explanations(caption_data=image_caption_w_blip2.output_caption_data[key][0],
                                                              data=self.data,
                                                              answer_column="multiple_choice_answer")
            self.data[f"prediction_explanation_{index+1}_w_gt"] = explanations
        # generate explanation generation for above generated predictions
        print("explanation generation for (question, predicted-answer, caption) triplet")
        self.data[f"prediction_explanation_w_pred"] = self.batch_processing_explanations(caption_data=image_caption_w_blip2.output_caption_data["output_3"][0],
                                                                                         data=self.data,
                                                                                         answer_column="prediction")
        
    def load_flan_t5(self, model_type: flan_model_types_="xxl"):
        model_type_mapping = {
            "xl": "google/flan-t5-xl",
            "xxl": "google/flan-t5-xxl"
        }
        # load model & tokenizer 
        # TODO: change the cache-dir according to your preference or remove it!
        self.model = T5ForConditionalGeneration.from_pretrained(model_type_mapping[model_type], device_map="auto", cache_dir="/home/users/sselvaraj/project/huggingface_hub/")
        self.tokenizer = T5Tokenizer.from_pretrained(model_type_mapping[model_type], cache_dir="/home/users/sselvaraj/project/huggingface_hub/")
        
    def rephrase_blip2_captions(self, example: pd.Series):
        """
        process already generated caption by Blip2 model architecture.
        Apply paraphrasing/summarization to derived caption-point for 
        each instance.
        """
        # decision-string 1: identify color-type
        color_category = example["color_category"]
        decision_string_1 = f"It is a {color_category} and includes colors like {example['color_listing']}" if color_category == "Color Image" else f"It is a {color_category}"
        # decision-string 2: identify location-setting
        location_setting = "outdoor_setting" if example["locations"] == "outside" else "indoor_setting"
        decision_string_2 = f"It appear to be situated in a {example['locations']} location. The environmental setting appears to be {example[location_setting]}"
        # decision-string 3: identify human-presence
        human_presence = example["human_presence"]
        decision_string_3 = f"There are {example['human_count']} humans in the scence. Their attire constitutes of {example['human_attire']}. They seems to be {example['age']}" if human_presence == "yes" else "It has no human presence."
        prompt = f"""Task: Paraphrase the input text, and make it grammatically correct.
    Input: In the image {example['main_caption']}. The following objects are present {example['object_listing']}. The standout features of the image is {example['standout_features']}. {decision_string_1}. The emotional ambiance of the scene is {example['emotional_ambiance']}. {decision_string_2}. The activity is {example['activity']}. {decision_string_3}. The camera perspective of the image is {example['camera_perspective']}"""
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids, max_length=200, min_length=20, temperature=0.2, top_k=0.2, num_beams=5)
        paraphrased_captions = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return paraphrased_captions.strip()

    def generate_predictions(self, example: pd.Series, question: str):
        paraphrased_captions = self.rephrase_blip2_captions(example)
        prompt = f"""Task: Question-Answering. Answer the question based on the provided context.
    Caption: {paraphrased_captions}
    Question: {question}
    Answer: 
    """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids)
        predicted_answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True) 
        return predicted_answer.strip()

    def generate_explanations(self, example: pd.Series, question: str, answer: str):
        paraphrased_captions = self.rephrase_blip2_captions(example)
        exp_prompt = f"""Task: Explain the reason behind answer to the question using the context to the image.
    Question: {question}
    Answer: {answer}
    Caption: {paraphrased_captions}
    Question: Why?
    Answer: 
    """
        input_ids = self.tokenizer(exp_prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids)
        predicted_explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return predicted_explanation.strip()

    def batch_processing_prediction(self, caption_data: pd.DataFrame, data: pd.DataFrame):
        predictions = []
        for index, example in tqdm(caption_data.iterrows(), total=100):
            prediction = self.generate_predictions(example, data.iloc[index]["question"])
            predictions.append(prediction)
        return predictions

    def batch_processing_explanations(self, caption_data: pd.DataFrame, 
                                      data: pd.DataFrame, answer_column: answer_columns_="multiple_choice_answer"):
        explanations = []
        for index, example in tqdm(caption_data.iterrows(), total=100):
            explanation = self.generate_explanations(example, data.iloc[index]["question"], data.iloc[index][answer_column])
            explanations.append(explanation)
        return explanations
    
    def save_data(self):
        path_to_file = os.path.join("./data", self.config['PATH_TO_SAMPLED_PREPROCESSED_DATA_FINAL'])
        self.data.to_csv(path_to_file, index=False)
        print(f"sampled-preprocessed-data w/ captions saved @ loc: {path_to_file}")