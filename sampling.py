from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sys import exit


PROJECTS = ["cassandra", "elasticsearch", "flink", "HBase", "JMeter", "kafka", "karaf", "wicket", "Zookeeper"]

log_level_map = {"trace": 0, "debug": 1, "info": 2, 'warn': 3, 'warning': 3, "error": 4, "fatal": 5}

test_size = 0.6

def print_distribution(df):
    print("Occurrences of each log_level:")
    print(df['log_level'].value_counts())



def random_n_shots(df, shot, project):

    sampled_df = df.sample(n=shot)
    print_distribution(sampled_df)
    sampled_df['instruction'] = "Between debug, warn, error, trace, and info, choose a suitable log level for the source code provided. "
    sampled_df['input'] = 'The source code is """' + sampled_df["code"] + '""", and the log message is ' + sampled_df["constant"] + '.'
    sampled_df['output'] = "The log level is " + sampled_df['log_level'] + "."
    sampled_df.to_json(f"logs/{project}_instructions_logs_train_{shot}.json", orient="records", indent=2)

    sampled_df['input'] = 'The previous method is """' + sampled_df["previousMethod"] + '""", the source code is """' + sampled_df["code"] + '""", and the log message is ' + sampled_df["constant"] + '.'
    sampled_df.to_json(f"logs/{project}_instructions_logs_train_{shot}_pm.json", orient="records", indent=2)

    return sampled_df

###need a separate function to generate ultimate because getting data from training sets only
def generate_ultimate():

    for project in PROJECTS:

        print(f"====={project}=====")

        ft_projects = PROJECTS.copy()
        ft_projects.remove(project)

        df = pd.DataFrame(columns=["index", "callsite", "log_level", "constant", "code"])
        for ft_project in ft_projects:
            df_project = pd.read_json(f"logs/{ft_project}_instructions_logs_train_0.6.json")
            df_project = df_project.drop(columns=[col for col in df_project if col not in ["index", "callsite", "log_level", "constant", "code"]])
            df = pd.concat([df, df_project])

        df['log_level'] = df['log_level'].str.lower()
        df['log_level'] = df['log_level'].replace('warning', 'warn')
        df = df[df['log_level'] != 'fatal']

        try:
            x_train, x_remain = train_test_split(df, test_size=test_size, stratify=df['log_level'])
        except ValueError:
            x_train, x_remain = train_test_split(df, test_size=test_size)
            stratified_train = 'Not Stratified'

        x_train['instruction'] = "Between debug, warn, error, trace, and info, choose a suitable log level for the source code provided. "
        x_train['input'] = 'The source code is """' + x_train["code"] + '""", and the log message is ' + x_train["constant"] + '.'
        x_train['output'] = "The log level is " + x_train['log_level'] + "."

        print_distribution(x_train)
        x_train.to_json(f"logs/{project}_instructions_logs_train_ultimate.json", orient="records", indent=2)

def generate_for_each_project():

    for project in PROJECTS:

        print(f"====={project}=====")

        df = pd.read_json(f"data/data-{project}.json")
        df = df[df['log_level'] != 'fatal']
        df['log_level'] = df['log_level'].str.lower()
        df['log_level'] = df['log_level'].replace('warning', 'warn')

        print_distribution(df)
        stratified_train = 'Train Stratified'

        try:
            x_train, x_remain = train_test_split(df, test_size=test_size, stratify=df['log_level'])
        except ValueError:
            x_train, x_remain = train_test_split(df, test_size=test_size)
            stratified_train = 'Not Stratified'

        stratified_test = 'Test Stratified'

        try:

            x_validate, x_test = train_test_split(x_remain, test_size=0.5, stratify=df['log_level'])
        except ValueError:
            x_validate, x_test = train_test_split(x_remain, test_size=0.5)
            stratified_test = 'Test Not Stratified'

        print(stratified_train)
        print(stratified_test)

        x_train.to_json(f"logs/{project}_logs_train_{test_size}.json", orient="records", indent=2)
        x_validate.to_json(f"logs/{project}_logs_validate_{test_size}.json", orient="records", indent=2)
        x_test.to_json(f"logs/{project}_logs_test_{test_size}.json", orient="records", indent=2)

        print_distribution(x_train)

        sampled_df = random_n_shots(x_train, 30, project)

        sampled_df = random_n_shots(sampled_df, 20, project)

        sampled_df = random_n_shots(sampled_df, 10, project)

        sampled_df = random_n_shots(sampled_df, 5, project)

generate_for_each_project()
generate_ultimate()