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

## Reference

1. DH. Park, LA. Hendricks, Z. Akata, A. Rohrbach, B. Schiele, T. Darrell, M. Rohrbach, Multimodal Explanations: Justifying Decisions and Pointing to the Evidence. in CVPR, 2018.
2. Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
