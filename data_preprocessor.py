import ast
import pandas as pd
from collections import defaultdict
from functools import reduce
from utils import load_from_json

# preprocess answer/annotations (useful when deriving prediction and saving data)
def process_annotations_data(data: pd.DataFrame):
    def compute_average(values):
        counts = {
            'yes': values.count('yes'),
            'no': values.count('no'),
            'maybe': values.count('maybe')
        }
        max_count = max(counts.values())
        most_common = [k for k, v in counts.items() if v == max_count]
        return most_common[0] if len(most_common) == 1 else 'maybe'
    # Convert the string representation to tuples
    for index in range(0, 10):
        data[f"answer_{index+1}"] = data[f"answer_{index+1}"].apply(lambda row: ast.literal_eval(row))
    annotation_columns = [f"answer_{index+1}" for index in range(0, 10)]
    overall_answers, overall_confidences = [], []
    for _, row in data[annotation_columns].iterrows():
        answers, confidences = zip(*row.tolist()) # extract answers and confidences as list
        overall_answer, overall_confidence = compute_average(answers), compute_average(confidences)
        overall_answers.append(overall_answer)
        overall_confidences.append(overall_confidence)
    # append overall annotations to the data
    data["overall_answer"] = overall_answers
    data["overall_confidence"] = overall_confidences
    data.drop(annotation_columns, axis=1, inplace=True)
    return data    

class Data_Preprocessor:
    def __init__(self, config):
        super(Data_Preprocessor, self).__init__()

        self.config = config
        # prepare validation-questions
        self.prepare_questions()
        # prepare validation-annotations
        self.prepare_annotations()     
        # prepare validation-explanations
        self.prepare_explanations()
        # merge and drop data supervised by respective explanations
        self.merge_and_drop_data()       
        # sample from already processed data
        self.sample_from_data()

    def prepare_questions(self):
        val_questions = load_from_json(path_to_file=self.config["PATH_TO_VALIDATION_QUESTIONS"])
        val_questions_df = pd.DataFrame(val_questions["questions"])
        val_questions_df["question_id"] = val_questions_df.apply(lambda row: str(row["question_id"]), axis=1)
        self.val_question_df = val_questions_df

    def prepare_annotations(self):
        val_annotations = load_from_json(path_to_file=self.config["PATH_TO_VALIDATION_ANNOTATIONS"])
        val_annotations_dict = defaultdict(list)
        for annotation in val_annotations["annotations"]:
            val_annotations_dict["question_id"].append(str(annotation["question_id"]))
            val_annotations_dict["question_type"].append(annotation["question_type"])
            val_annotations_dict["answer_type"].append(annotation["answer_type"])
            val_annotations_dict["multiple_choice_answer"].append(annotation["multiple_choice_answer"])
            for ans_index, answer in enumerate(annotation["answers"]):
                val_annotations_dict[f"answer_{ans_index+1}"].append(tuple(list(answer.values())[:-1]))
            val_annotations_dict["image_id"].append(annotation["image_id"])
        val_annotations_df = pd.DataFrame(val_annotations_dict)
        self.val_annotations_df = val_annotations_df

    def prepare_explanations(self):
        val_explanations = load_from_json(path_to_file=self.config["PATH_TO_VALIDATION_EXPLANATIONS"])
        val_explanations_dict = defaultdict(list)
        for question_id, explanations in val_explanations.items():
            val_explanations_dict["question_id"].append(question_id)
            val_explanations_dict["explanation_1"].append(explanations[0])
            val_explanations_dict["explanation_2"].append(explanations[1])
            val_explanations_dict["explanation_3"].append(explanations[2])
        val_explanation_df = pd.DataFrame(val_explanations_dict)
        self.val_explanation_df = val_explanation_df

    def merge_and_drop_data(self):
        df_list = [self.val_question_df, self.val_annotations_df, self.val_explanation_df]
        self.data = reduce(lambda left, right: pd.merge(left, right, on=['question_id'], how='outer'), df_list).dropna(axis=0).reset_index(drop=True)
        self.data.drop(["image_id_y"], axis=1, inplace=True)
        self.data.rename(columns={
            "image_id_x": "image_id"
        }, inplace=True)

    def sample_from_data(self, k: int=100):
        df = self.data
        # distribute all `question_type` w/ n=1 from the original data
        distribute_sampled_df = df.groupby("question_type").apply(lambda x: x.sample(n=1)).reset_index(drop=True)
        # randomly sample the remaning rows
        random_sampled_df = df[~df["question_id"].isin(distribute_sampled_df["question_id"])].sample(n=k-len(distribute_sampled_df))
        # concatenate the data
        self.sample_data = pd.concat([distribute_sampled_df, random_sampled_df]).reset_index(drop=True)

    def save_data(self):
        # save preprocessed data
        self.data.to_csv(self.config["PATH_TO_PREPROCESSED_DATA"], index=False)
        print(f"preprocessed-data saved @ loc: {self.config['PATH_TO_PREPROCESSED_DATA']}")
        # save sampled preprocessed data
        self.sample_data.to_csv(self.config["PATH_TO_SAMPLED_PREPROCESSED_DATA"], index=False)
        print(f"sampled-preprocessed-data saved @ loc: {self.config['PATH_TO_SAMPLED_PREPROCESSED_DATA']}")