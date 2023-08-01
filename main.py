import pandas as pd
from utils import config
from utils import download_and_extract_data
from data_preprocessor import Data_Preprocessor
from image_caption_w_blip2 import Image_Caption_w_Blip2
from predict_and_explain_w_flant5 import Predict_and_Explain_w_FlanT5
from auto_metrics import batch_processing_auto_metrics

if __name__ == "__main__":
    if config["DOWNLOAD_DATA"]:
        # download VQA-v2 data from the original data-repository [NOTE: VQA-x data is already stored under ./data/textual/]
        _ = [download_and_extract_data(data_type=data_type, url=url) for data_type, url in zip(config["DATA_TYPES"], config["DATA_TYPE_URLS"])]
    if config["PREPROCESS_DATA"]:
        # configure data, perform data processing and sampling over varied question_types
        data_preprocessor = Data_Preprocessor(config=config)
        data_preprocessor.save_data()
    if config["PREDICT_AND_EXPLAIN"]:
        # generate captions using blip2 model architecture
        # generate answers as prediction using the flant5xxl model & {question, caption} pairs.
        # generate explanations using the flant5xxl model & {question, answer, caption} triplet with answers as 
        # either multiple-choice answer (gt) or predicted answer 
        predict_and_explain_w_flant5 = Predict_and_Explain_w_FlanT5(config=config)
        predict_and_explain_w_flant5.save_data()
        data = predict_and_explain_w_flant5.data
    else:
        # load aleady cached data w/ captions, prediction, & explanations
        data = pd.read_csv(config["PATH_TO_SAMPLED_PREPROCESSED_DATA_FINAL"])
    # calculate METEOR, ROUGE & BLEU scores
    output_results = batch_processing_auto_metrics(data=data)
    print(output_results) 