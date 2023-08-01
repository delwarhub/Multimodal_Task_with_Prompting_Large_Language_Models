import evaluate
import pandas as pd
from typing import Literal
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = 12, 8

metric_types_ = Literal["meteor", "rouge", "bleu"]
explanation_columns_ = Literal["prediction_explanation_w_pred",
                               "prediction_explanation_1_w_gt",
                               "prediction_explanation_2_w_gt",
                               "prediction_explanation_3_w_gt"]

def calculate_meteor(predictions: list,
                     references: list):
            
    meteor_score = evaluate.load('meteor')
    result = meteor_score.compute(
        predictions=predictions,
        references=references
    ) 
    return result

def calculate_rouge(predictions: list,
                     references: list):
    
    rouge_score = evaluate.load('rouge')
    result = rouge_score.compute(
        predictions=predictions,
        references=references
    ) 
    return result

def calculate_bleu(predictions: list,
                   references: list):
    
    bleu_score = evaluate.load("bleu")
    result = bleu_score.compute(
        predictions=predictions,
        references=references
    ) 
    return result

def metric_mapper(data: pd.DataFrame, metric_type: metric_types_="meteor", explanation_column: explanation_columns_="prediction_explanation_w_pred"):
    metric_2_function_mapping = {
        "meteor": calculate_meteor,
        "rouge": calculate_rouge,
        "bleu": calculate_bleu
    }
    references = list(zip(data["explanation_1"].tolist(), data["explanation_2"].tolist(), data["explanation_3"].tolist()))
    predictions = data[explanation_column].tolist()
    results = metric_2_function_mapping[metric_type](predictions=predictions,
                                                     references=references)
    return results

def batch_processing_auto_metrics(data: pd.DataFrame):

    meteor_dict = defaultdict(list)
    rouge_1_dict = defaultdict(list)
    rouge_2_dict = defaultdict(list)
    rouge_L_dict = defaultdict(list)
    bleu_dict = defaultdict(list)
    for explanation_column in ["prediction_explanation_w_pred", "prediction_explanation_1_w_gt", "prediction_explanation_2_w_gt","prediction_explanation_3_w_gt"]:
        for metric_type in ["meteor", "rouge", "bleu"]:
            result = (metric_mapper(data=data,
                                    metric_type=metric_type,
                                    explanation_column=explanation_column))

            if metric_type == "meteor":
                meteor_dict[explanation_column].append(result["meteor"])
            elif metric_type == "rouge":
                rouge_1_dict[explanation_column].append(result["rouge1"])
                rouge_2_dict[explanation_column].append(result["rouge2"])
                rouge_L_dict[explanation_column].append(result["rougeL"])
            elif metric_type == "bleu":
                bleu_dict[explanation_column].append(result["bleu"])
            
    results = {"meteor": meteor_dict, "rouge1": rouge_1_dict, "rouge2": rouge_2_dict, "rougeL": rouge_L_dict, "bleu": bleu_dict}

    # plot graphs (barplots)
    # def plot_rouge(results):
    meteor_val = [item[0] for item in list(results["meteor"].values())]
    rouge1_val = [item[0] for item in list(results["rouge1"].values())]
    rouge2_val = [item[0] for item in list(results["rouge2"].values())]
    rougeL_val = [item[0] for item in list(results["rougeL"].values())]
    bleu_val = [item[0] for item in list(results["bleu"].values())]
    columns = ['explanation_w_pred', 'explanation_w_gt_1', 'explanation_w_gt_2', 'explanation_w_gt_3']
    result_data = pd.DataFrame({
        "explanation_type": columns,
        "meteor_score": meteor_val,
        "rouge1_score": rouge1_val,
        "rouge2_score": rouge2_val,
        "rougeL_score": rougeL_val,
        "bleu_score": bleu_val
    })
    sns.barplot(data=result_data, x="explanation_type", y="meteor_score")
    plt.savefig("./plots/meteor_plot.png")
    sns.barplot(data=result_data, x="explanation_type", y="rouge1_score")
    plt.savefig("./plots/rouge1_plot.png")
    sns.barplot(data=result_data, x="explanation_type", y="rouge2_score")
    plt.savefig("./plots/rouge2_plot.png")
    sns.barplot(data=result_data, x="explanation_type", y="rougeL_score")
    plt.savefig("./plots/rougeL_plot.png")
    sns.barplot(data=result_data, x="explanation_type", y="bleu_score")
    plt.savefig("./plots/bleu.png")

    print("ploted automatic-metrics graphs succesfully under directory ./plots")