# XAI Project 4
## Project 4 - Multimodal Task with Prompting Large Language Models

## Task Data: VQA-X

The VQA dataset is a popular benchmark dataset for evaluating Visual Question Answering systems. It was created to promote research in the area of combining computer vision and natural language processing. The dataset contains images with associated questions and corresponding answers, making it suitable for training and evaluating models that can understand both visual content and textual questions. 

The VQA dataset includes real-world images from the MS COCO (Microsoft Common Objects in Context) dataset and abstract scenes from the Abstract Scenes dataset. Each image is paired with multiple questions, and each question has ten different answers collected from human annotators. This diversity of answers allows the evaluation of models based on open-ended and multiple-choice questions.

## Training & Testing Set-Up  (Linux SetUp)

1. Clone the repository

```
git clone [git clone https URL]
```

2. Create a Python virtual environment

```
# Update and upgrade
sudo apt update
sudo apt -y upgrade

# check for python version
python3 -V

# install python3-pip
sudo apt install -y python3-pip

# install-venv
sudo apt install -y python3-venv

# Create virtual environment
python3 -m venv my_env


# Activate virtual environment
source my_env/bin/activate
```

3. Install project dependent files

```
pip install requirements.txt
```

4. Run main.py

```
python3 main.py
```

## Download VQA-v2 & VQA-X Dataset 

```
# edit ./config.yaml file
vim ./config.yaml

# change following attributes
DOWNLOAD_DATA: True

# download dataset (commands already provided)
python3 main.py
```

## Preprocess VQA-X & Sample 100 Instances

```
# edit ./config.yaml file
vim ./config.yaml

# change following attributes 
PREPROCESS_DATA: True

# post-process VQA-X dataset and perform sampling
python3 main.py
```

## Image Captioning, Prediction & Explanation Generation

```
# edit ./config.yaml file
vim ./config.yaml

# change following attributes
PREDICT_AND_EXPLAIN: True

# applying captioning procedure using blip2 model architecture
# predict and explain using the flant5xxl
python3 main.py
```

# Project Directory Tree

```
└── Project4/
    ├── data/
    │   ├── vqa_x_data.csv
    │   ├── textual/
    |   |   ├── sample_esnlive_emotion_oriented.csv
    |   |   └── sample_esnlive_topic_oriented.csv
    │   ├── Questions/
    |   |   ├── v2_Annotations_Val_mscoco.zip
    |   |   └── v2_mscoco_val2014_annotations.json
    |   ├── Annotations/
    |   |   ├── v2_Annotations_Val_mscoco.zip
    |   |   └── v2_mscoco_val2014_annotations.json
    |   ├── Images/
    |   |   ├── val2014.zip
    |   |   └── val2014/
    │   ├── vqa_x_sampled_data.csv
    |   └── vqa_x_sampled_data_final.csv
    ├── huggingface_hub/
    ├── utils.py
    ├── data_preprocessor.py
    ├── config.yaml
    ├── auto_metrics.py
    ├── main.py
    ├── image_caption_w_blip2.py
    ├── predict_and_explain_w_flant5.py
    ├── README.md
    └── requirements.txt
```

``` Note: Please install salesforce-lavis separately, due to some issues with environment configuration we were not able to install it locally.```

##Tasks
1.	Pick one of the following multimodal tasks: VQA-X, E-SNLI-VE, A-OKVQA, VCR
2.	Select 100 samples from any split of the dataset (distributed equally across class labels, when possible)
3.	Use image captioning to obtain textual representations for visual content (BLIP-2 or Instruct-BLIP for captioning)
4.	Prompt a large language model (FLAN-T5 XXL model or any other model capable of generating rationale) to solve the task. You can use the prompt templates from a recent paper (see Appendix) and adjust it for your selected task. You can refer to this document on learning what prompting is about. You can refer to this leaderboard on deciding for which open access large language model to choose.
5.	Prompt the model to first generate the task-specific prediction (text) and then prompt it again to generate an explanation to the prediction. After appending the model prediction to the prompt text, simply add something like "Why is that? Generate an explanation" (or any text that instructs the model to explain why it generated the prediction (X) for the task) and prompt the model again to generate a rationale. Another possible way is to prompt the language model to generate the required task prediction and the related explanation in one step (by adding this instruction into the prompt).
6.	Compare generated explanations with ground truth using the metrics (BLEU, ROUGE, METEOR etc.) that are used in the selected dataset and its paper. One metric is sufficient.
7.	Evaluate the generated explanations for the 100 samples manually (where each member annotates the same sample and you take a look at the average across the team) by following an annotation scheme from the literature (check how other methods evaluated the generated explanation and how users rated them, based on what aspects, e.g. usefulness, clarity of the explanation, etc.). You can refer to the survey papers that you read previously.
8.	Present the overall findings of the prompting strategy, annotation scheme and analysis of the generated explanations (show good & bad predictions/explanations).
9.	You can run the selected language model on the Colab Notebook (usually requires GPU with VRAM of 25-30 GB), apply for the university account (as described in the forum before but you need to specify that you need access to A100 GPU with 80GB), or via HuggingFace API. Alternatively, you can also look for quanitzed versions of the language models that lets you load them into smaller memory machines or even running them directly on CPU.
10.	Alternatively, you can also use GPT models for this project.


## Reference

1. DH. Park, LA. Hendricks, Z. Akata, A. Rohrbach, B. Schiele, T. Darrell, M. Rohrbach, Multimodal Explanations: Justifying Decisions and Pointing to the Evidence. in CVPR, 2018.
2. Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
