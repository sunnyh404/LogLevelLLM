
import re
import os
import torch
import datasets
import transformers
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys
import time
import socket
import os
import sys
import pickle
from sys import exit

hostname = socket.gethostname()
main_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

project = sys.argv[1]

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
batch_size = int(sys.argv[2])
num_class = 4 #will be dynamically regenerated based on number of log levels
max_seq_l = 512
lr = 1e-5
num_epochs = 10 #original value = 5
use_cuda = True
model_name = sys.argv[3]  #expected value "roberta"
pretrainedmodel_path = sys.argv[4] #expected value "roberta-base"
test_size = sys.argv[5] #expected value "0.9"
ft_project = sys.argv[6]
pm = ""

level_labels = [['trace'], ['debug'], ['info'], ['warn'], ['error']]

log_level_map = {"trace": 0, "debug": 1, "info": 2, 'warn': 3, 'warning': 3, "error": 4}

redundant_columns = ['index', 'constant', 'code', 'log_label']

projects = []

if project == "all":
    projects = ["cassandra", "elasticsearch", "flink", "HBase", "JMeter", "kafka", "karaf", "wicket", "Zookeeper"]
else: projects.append(project)

for project in projects:

    arguments = "_".join([project, ft_project, model_name.replace("/", ""), os.path.basename(pretrainedmodel_path), str(test_size)]) + pm
    output_file = f"results/output_{arguments}_x.txt"

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'w') as f:
        print("Project: " + project, file=f)
        print("ft_project: " + ft_project, file=f)
        print("model_name: " + model_name, file=f)
        print("pretrainedmodel_path: " + os.path.basename(pretrainedmodel_path), file=f)
        print("test_size: " + str(test_size), file=f)
        print("input: code, constant", file=f)


        if str(test_size) in ("0", "5"):
            train_path = f"logs/{project}_instructions_logs_train_0.6{pm}.json"
        else:
            if str(test_size) == "0.5":
                train_path = f"logs/{project}_instructions_logs_train_5{pm}.json"
            else:
                train_path = f"logs/{ft_project}_instructions_logs_train_{str(test_size)}{pm}.json"

        if float(test_size) >= 0:
            validate_path = f"logs/{project}_logs_validate_0.6{pm}.json"
            test_path = f"logs/{project}_logs_test_0.6{pm}.json"
        else:
            validate_path = f"logs/{project}_logs_validate_{str(test_size)}{pm}.json"
            test_path = f"logs/{project}_logs_test_{str(test_size)}{pm}.json"


        train_dataset = pd.read_json(train_path)

        validate_dataset = pd.read_json(validate_path)

        train_dataset['log_label'] = train_dataset['log_level'].str.lower().map(log_level_map)
        validate_dataset['log_label'] = validate_dataset['log_level'].str.lower().map(log_level_map)

        train_dataset = train_dataset.dropna(subset=['log_label'])
        validate_dataset = validate_dataset.dropna(subset=['log_label'])

        train_dataset = train_dataset.drop(columns=[col for col in train_dataset if col not in redundant_columns])
        validate_dataset = validate_dataset.drop(columns=[col for col in validate_dataset if col not in redundant_columns])

        train_dataset = train_dataset.applymap(str)
        validate_dataset = validate_dataset.applymap(str)

        # processing and splitting the dataset
        train_dataset = Dataset.from_dict(train_dataset)
        validate_dataset = Dataset.from_dict(validate_dataset)

        train_val_test = {}
        train_val_test['train'] = train_dataset
        train_val_test['validation'] = validate_dataset

        from openprompt.data_utils import InputExample
        dataset = {}
        for split in ['train', 'validation']:
            dataset[split] = []
            for data in train_val_test[split]:

                if pm == "_pm":
                    input_example = InputExample(guid = data["index"], text_a = data["pm_function_sans_tag"], text_b = data["block_till_log_sans_tag"], meta = data["log_message"], label=int(float(data['log_label'])))
                else:
                    input_example = InputExample(guid = data["index"], text_a = data["code"], text_b = data["constant"], label=int(float(data['log_label'])))

                dataset[split].append(input_example)


        shot5 = ""

        if str(test_size) == "5":

            df_5shot = pd.read_json(f"logs/{project}_instructions_logs_train_5{pm}.json")
            df_5shot['log_label'] = df_5shot['log_level'].str.lower().map(log_level_map)

            for index, row in df_5shot.iterrows():
                if pm == "_pm":
                    shot5 = shot5 + f"The previous method is {row['pm_function_sans_tag']}, the source code is {row['block_till_log_sans_tag']}, and the log message is {row['log_message']}. The log level is {row['log_label']}."
                else:
                    shot5 = shot5 + f'The source code is """' + row['code'] + '""", and the log message is ' + row['constant'] + '. The log level is ' + row['log_level'] + '.'

        pattern = r'[^a-zA-Z0-9 ]'
        shot5 = re.sub(pattern, '', shot5)
        shot5 = re.sub(' +', ' ', shot5)

        # load plm
        from openprompt.plms import load_plm
        plm, tokenizer, model_config, WrapperClass = load_plm(model_name, pretrainedmodel_path)

        # construct hard template
        from openprompt.prompts import ManualTemplate


        if pm == "_pm":
            template_text = 'The previous method is {"placeholder":"text_a", "shortenable": True}, the source code is  {"placeholder":"text_b", "shortenable": True}, and the log message is  {"placeholder":"meta", "shortenable": True}. The log level is {"mask"}. '
        else:
            template_text = 'Between debug, warn, error, trace, and info, choose a suitable log level for the source code provided. ' + shot5 + 'The source code is {"placeholder":"text_a", "shortenable": True}. the log message is {"placeholder":"text_b", "shortenable": True}. The log level is {"mask"}. '

        mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)

        # define the verbalizer
        from openprompt.prompts import ManualVerbalizer
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=num_class, label_words=level_labels)

        # define prompt model for classification
        from openprompt import PromptForClassification
        prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
        if use_cuda:
            prompt_model=  prompt_model.cuda()

        # DataLoader
        from openprompt import PromptDataLoader
        train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, batch_size=batch_size,shuffle=False,
            teacher_forcing=False, predict_eos_token=False, truncate_method="head", shortenable=True)
        validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, batch_size=batch_size,shuffle=False,
            teacher_forcing=False, predict_eos_token=False, truncate_method="head", shortenable=True)

        from transformers import  AdamW, get_linear_schedule_with_warmup

        loss_func = torch.nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        # it's always good practice to set no decay to biase and LayerNorm parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in prompt_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in prompt_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        num_training_steps = num_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        def test(prompt_model, test_dataloader, run_type):
            prompt_model.eval()

            allpreds = []
            alllabels = []
            with torch.no_grad():
                for step, inputs in enumerate(test_dataloader):
                    if use_cuda:
                        inputs = inputs.cuda()
                    logits = prompt_model(inputs)
                    labels = inputs['label']
                    alllabels.extend(labels.cpu().tolist())
                    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                acc = accuracy_score(alllabels, allpreds)
                precisionwei, recallwei, f1wei, _ = precision_recall_fscore_support(alllabels, allpreds, average='weighted')
                precisionmi, recallmi, f1mi, _ = precision_recall_fscore_support(alllabels, allpreds, average='micro')
                precision, recall, f1, _ = precision_recall_fscore_support(alllabels, allpreds, average=None)

                print("\nproject {} ;Epoch: {}; runtype: {} ; labels: {}".format(project, epoch, run_type, str(alllabels)), flush=True, file=f)
                print("\nproject {} ;Epoch: {}; runtype: {} ; predictions: {}".format(project, epoch, run_type, (allpreds)), flush=True, file=f)
                print("\nproject: {} ; runtype: {} ;acc: {} ;weighted-f1: {}  ;micro-f1: {}".format(project, run_type, acc, f1wei, f1mi), file=f)
            return acc, f1wei, f1mi

        from tqdm.auto import tqdm

        progress_bar = tqdm(range(num_training_steps))
        bestmetric = 0
        bestepoch = 0
        for epoch in range(num_epochs):
            prompt_model.train()

            tot_loss = 0

            if str(test_size) not in ("0", "5"):
                for step, inputs in enumerate(train_dataloader):
                    if use_cuda:
                        inputs = inputs.cuda()
                    logits = prompt_model(inputs)
                    labels = inputs['label']
                    loss = loss_func(logits, labels)
                    loss.backward()
                    tot_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    progress_bar.update(1)
                print("\nproject {} ;Epoch: {}; average loss: {} ".format(project, epoch, tot_loss/(step+1)), flush=True, file=f)
            acc, f1wei, f1mi = test(prompt_model, validation_dataloader, "validation")

        f.close()
