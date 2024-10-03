from sys import exit
import sys
import os
import pandas as pd
import re
import ast
import copy
import calculations_util


projects = ["cassandra", "elasticsearch", "flink", "HBase", "JMeter", "kafka", "karaf", "wicket", "Zookeeper"]
log_levels = ["warn", "trace", "info", "error", "debug"]

tokenizer = ["roberta"]
pretrainedmodels = ["graphcodebert-base"]

pretrained_to_model = {pretrained: model for pretrained, model in zip(pretrainedmodels, tokenizer)}

special_llms = ["CodeLlama-7b-hf"]

llms = pretrainedmodels + special_llms

shots = ["0", "5", "0.5", "10", "20", "30", "0.6"]


def remove_before_sixth_Input_occurrence(input_string):
    count = 0
    index = 0
    while count < 6:
        index = input_string.find("Input:", index + 1)
        if index == -1:
            break
        count += 1

    if count < 6:
        # Less than six occurrences found, return original string
        return input_string

    # Find the index of the first character of the string to remove
    index_to_remove = input_string.rfind("\n", 0, index) + 1

    # Return the modified string
    return input_string[index_to_remove:]

def textgen_return_level_prediction(prediction, shot):
    if shot == "5":
        prediction = remove_before_sixth_Input_occurrence(prediction)

    level_prediction = "N/A"

    pattern = r"### Response:(.*?)\nThe log level(.*?)(is|should be|is set to|should be set to) (debug|warn|info|trace|error|fatal)"
    compiled_pattern = re.compile(pattern, re.IGNORECASE)
    match1 = compiled_pattern.search(prediction)

    # Check if the match is found

    if match1:
        level_prediction = match1.groups()[3].lower().strip()

    else:
        pattern = r"### Response:(.*?)The log level(.*?)(is|should be|is set to|should be set to) (debug|warn|info|trace|error|fatal)"
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        match1 = compiled_pattern.search(prediction)

        if match1:
            level_prediction = match1.groups()[3].lower().strip()
        else:
            pattern = r"Solution:(.*?)\nThe log level(.*?)(is|should be|is set to|should be set to) (debug|warn|info|trace|error|fatal)"
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            match2 = compiled_pattern.search(prediction)

            if match2:
                level_prediction = match2.groups()[3].lower().strip()
            else:
                pattern = r"Solution:(.*?)\nThe log level(.*?)(is|should be|is set to|should be set to) (debug|warn|info|trace|error|fatal)"
                compiled_pattern = re.compile(pattern, re.IGNORECASE)
                match2 = compiled_pattern.search(prediction)

                if match2:
                    level_prediction = match2.groups()[3].lower().strip()
                else:

                    pattern = r"The logger is configured to log at (debug|warn|info|trace|error|fatal) level\."
                    compiled_pattern = re.compile(pattern, re.IGNORECASE)
                    match3 = compiled_pattern.search(prediction)

                    if match3:
                        level_prediction = match3.groups()[0].lower().strip()

                    else:
                        pattern = r"Solution:(.*?)\nThe response (is|should be|is set to|should be set to)(.*?)(debug|warn|info|trace|error|fatal)"
                        compiled_pattern = re.compile(pattern, re.IGNORECASE)
                        match4 = compiled_pattern.search(prediction)

                        if match4:
                            level_prediction = match4.groups()[3].lower().strip()
                        else:
                            pattern = r"Solution:(.*?)The response (is|should be|is set to|should be set to)(.*?)(debug|warn|info|trace|error|fatal)"
                            compiled_pattern = re.compile(pattern, re.IGNORECASE)
                            match4 = compiled_pattern.search(prediction)

                            if match4:
                                level_prediction = match4.groups()[3].lower().strip()

    return level_prediction

def getLabelsPredictions(shot, remove_str, split_string):
    target = ""
    if shot == "0" or shot == "5":
        pattern = f"Epoch: 0; runtype: validation ; "
    else:
        pattern = f"Epoch: 9; runtype: validation ; "

    for item in split_string:
        if pattern in item and remove_str in item:
            target = item
            break

    if target != "":
        index = target.find(remove_str)

        if index > 0:
            target = target[index + len(remove_str):].strip()
            target = ast.literal_eval(target)
    else: target = ""
    return target

def fillmask_return_level_prediction(project, shot, result_file):

    with open(result_file) as f:

        split_string = f.readlines()

        label_line = getLabelsPredictions(shot, "labels:", split_string)
        prediction_line = getLabelsPredictions(shot, "predictions:", split_string)

        df_merged = pd.read_json(f"logs/{project}_logs_validate_0.6.json")
        df_merged = df_merged.drop(columns=[col for col in df_merged if col not in ['index', 'log_level']])

        df_merged["log_level"] = df_merged["log_level"].str.lower()
        df_merged['log_level'] = df_merged['log_level'].replace('warning', 'warn')
        df_merged["level_prediction"] = ""
        df_merged["project"] = ""

        for index, row in df_merged.iterrows():
            y = prediction_line[index]

            if y == "trace" or y == 0:
                y = "trace"
            elif y == "debug" or y == 1:
                y = "debug"
            elif y == "info" or y == 2:
                y = "info"
            elif y == "warn" or y == "warning" or y == 3:
                y = "warn"
            elif y == "error" or y == 4:
                y = "error"

            df_merged.at[index, 'level_prediction'] = y.strip()

    return df_merged[['index', 'project', 'log_level', 'level_prediction']]

#RQ1
def get_textgen_results():

    for llm in special_llms:
        for shot in shots:
            for project in projects:

                df_results = pd.read_json(f"logs/{project}_logs_validate_0.6.json")
                df_results = df_results[['index', 'log_level']]
                df_results["log_level"] = df_results["log_level"].str.lower()
                df_results['log_level'] = df_results['log_level'].replace('warning', 'warn')

                df_results['prediction'] = "empty"
                df_results['level_prediction'] = ""
                df_results['project'] = project

                for index, row in df_results.iterrows():

                    if row['prediction'] == "empty" or row['level_prediction'] == "":
                        ind_results_file = f"results/{llm}/{project}/{shot}/output_finetuned_validate_{project}_{llm}_{shot}_{row['index']}_pm.json"

                        if os.path.exists(ind_results_file) and os.path.getsize(ind_results_file) > 0:
                            ind_results = pd.read_json(ind_results_file)

                            if not ind_results.empty:

                                df_results.at[index, 'prediction'] = ind_results['prediction'].iloc[0]

                                if ind_results['prediction'].iloc[0] != "empty":

                                    df_results.at[index, 'level_prediction'] = textgen_return_level_prediction(ind_results['prediction'].iloc[0], shot)

                df_results[['index', 'project', 'log_level', 'level_prediction']].to_json(f"results/results_{llm}_{project}_{shot}_pm.json", orient="records", indent=2)


def get_fillmask_results():
    for llm in pretrainedmodels:
        for shot in shots:
            for project in projects:
                print(f"{llm}\t{shot}\t{project}")
                tokenizer = pretrained_to_model.get(llm)
                result_file = f"results/output_{project}_{tokenizer}_{llm}_{shot}_pm.txt"

                if os.path.exists(result_file):
                    df_results = fillmask_return_level_prediction(project, shot, result_file)
                    df_results[['index', 'project', 'log_level', 'level_prediction']].to_json(f"results/results_{llm}_{project}_{shot}_pm.json", orient="records", indent=2)

def calculate_metrics_llm_shot_project():

    print("Calculating calculate_metrics_llm_shot_project")

    columns = ["llm", "shot", "project", "Accuracy", "AUC", "AOD"]
    output = pd.DataFrame(columns=columns)

    for index, llm in enumerate(llms):
        for shot in shots:
            for project in projects:
                result_file = f"results/results_{llm}_{project}_{shot}_pm.json"
                if os.path.exists(result_file):
                    df = pd.read_json(result_file)

                    df = df[df['level_prediction'] != ""]

                    df['log_level'] = df['log_level'].str.lower()
                    df['level_prediction'] = df['level_prediction'].str.lower()

                    original_levels = df['log_level'].tolist()
                    predicted_levels = df['level_prediction'].tolist()
                    val_accuracy, val_auc_ovr_mic, val_oacc = calculations_util.getMetrics(original_levels, predicted_levels)

                    val_accuracy = "{:.2f}".format(val_accuracy * 100)
                    val_auc_ovr_mic = "{:.2f}".format(val_auc_ovr_mic * 100)
                    val_oacc = "{:.2f}".format(val_oacc * 100)

                    row = [llm, shot, project, val_accuracy, val_auc_ovr_mic, val_oacc]
                    output.loc[len(output.index)] = row
    print(output.to_csv(sep='\t', index=False))
    output.to_json("log_level_results_rq2.json", orient="records", indent=2)

get_fillmask_results()
get_textgen_results()
calculate_metrics_llm_shot_project()

def RQ2_table():
    df = pd.read_json("log_level_results_rq2.json", dtype={'shot': str})
    grouped_df = df.groupby(['llm', 'shot']).agg({
        'Accuracy': 'mean',
        'AUC': 'mean',
        'AOD': 'mean'
    }).reset_index()
    grouped_df[['Accuracy', 'AUC', 'AOD']] = grouped_df[['Accuracy', 'AUC', 'AOD']].round(2)

    print(grouped_df.to_csv(sep='\t', index=False))

RQ2_table() ###Results for RQ2