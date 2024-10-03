import pandas as pd



projects = ["cassandra", "elasticsearch", "flink", "HBase", "JMeter", "kafka", "karaf", "wicket", "Zookeeper"]



##FINETUNING PART
###Finetuning Text Gen models for RQ1
def RQ1_finetune():
    for llm in special_llms:
        shots = ["0.5", "10", "20", "30", "0.6"]
        for shot in shots:
            for project in projects:

                if shot == "5" or shot == "0.5":
                    data_path = f"logs/{project}_instructions_logs_train_5.json"
                else:
                    data_path = f"logs/{project}_instructions_logs_train_{shot}.json"

                print(f'''python finetune_codellama.py         --base_model "../{llm}"         --data_path "{data_path}"         --output_dir "saved_model/{llm}/{project}_{shot}"         --batch_size 2         --micro_batch_size 2         --num_epochs 10         --learning_rate 5e-4         --cutoff_len 2048         --val_set_size 0         --lora_r 8         --lora_alpha 16         --lora_dropout 0.05         --lora_target_modules '[q_proj,v_proj]'         --train_on_inputs         --group_by_length > "saved_model/{llm}/{project}_{shot}_finetune.txt"''')

###Finetuning Text Gen models for RQ2
def RQ2_finetune():
    for llm in special_llms:
        shots = ["0.5", "10", "20", "30", "0.6"]
        for shot in shots:
            for project in projects:

                if shot == "5" or shot == "0.5":
                    data_path = f"logs/{project}_instructions_logs_train_5_pm.json"
                else:
                    data_path = f"logs/{project}_instructions_logs_train_{shot}_pm.json"

                print(f'''python finetune_codellama.py         --base_model "../{llm}"         --data_path "{data_path}"         --output_dir "saved_model/{llm}/{project}_{shot}_pm"         --batch_size 2         --micro_batch_size 2         --num_epochs 10         --learning_rate 5e-4         --cutoff_len 2048         --val_set_size 0         --lora_r 8         --lora_alpha 16         --lora_dropout 0.05         --lora_target_modules '[q_proj,v_proj]'         --train_on_inputs         --group_by_length > "saved_model/{llm}/{project}_{shot}_finetune_pm.txt"''')


###Finetuning Text Gen models for RQ3 Enlarged
def RQ3_finetune_enlarged():
    for llm in special_llms:
        shots = ["0.6"]
        for shot in shots:
            for project in projects:
                print(f'''python finetune_codellama.py         --base_model "../{llm}"         --data_path "logs/{project}_instructions_logs_train_ultimate.json"    --output_dir "saved_model/{llm}/{project}_ultimate"         --batch_size 2         --micro_batch_size 2         --num_epochs 10         --learning_rate 5e-4         --cutoff_len 2048         --val_set_size 0         --lora_r 8         --lora_alpha 16         --lora_dropout 0.05         --lora_target_modules '[q_proj,v_proj]'         --train_on_inputs         --group_by_length > "saved_model/{llm}/{project}_ultimate_finetune.txt"
    ''')

##GENERATE RESULTS PART
###RQ1 Fill Mask generate results
def RQ1_FillMask_results():
    for llm in pretrainedmodels:
        for shot in shots:
            for project in projects:

                tokenizer = pretrained_to_model.get(llm)

                print(f'python fillmask_prompt.py "{project}" 8 "{tokenizer}" "../{llm}" "{shot}"')


###RQ1 Text Gen generate results
def RQ1_TextGen_results():
    for llm in special_llms:
        for shot in shots:
            for project in projects:

                ft_projects = projects.copy()
                ft_projects.remove(project)

                for ft_project in ft_projects:
                    df_logs = pd.read_json(f"../logs/{project}_logs_validate_0.6.json")
                    for index, row in df_logs.iterrows():
                        file_index = row['index']
                        print(f'python textgen_prompt.py "{project}" "{file_index}" "validate" "../{llm}" "0.6" ""   ')

###Generate commands to be used in RQ2
###RQ2 Fill Mask generate results
def RQ2_FillMask_results():
    for llm in pretrainedmodels:
        for shot in shots:
            for project in projects:

                tokenizer = pretrained_to_model.get(llm)

                print(f'python fillmask_prompt_rq2.py "{project}" 8 "{tokenizer}" "../{llm}" "{shot}"')


###RQ2 Text Gen generate results
def RQ2_TextGen_results():
    for llm in special_llms:
        for shot in shots:
            for project in projects:

                df_logs = pd.read_json(f"logs/{project}_logs_validate_0.6_pm.json")
                for index, row in df_logs.iterrows():
                    file_index = row['index']
                    print(f'python textgen_prompt_rq2.py "{project}" "{file_index}" "validate" "../{llm}" "0.6" ""   ')


###Generate commands to be used in RQ3
###RQ3 cross dataset
###RQ3 Fill Mask cross dataset
def RQ3_FillMask_Cross():
    for llm in pretrainedmodels:

        tokenizer = pretrained_to_model.get(llm)

        for shot in shots:
            for project in projects:

                ft_projects = projects.copy()
                ft_projects.remove(project)

                for ft_project in ft_projects:
                    print(f'python fillmask_prompt_x.py "{project}" 8 "{tokenizer}" "../{llm}" "{shot}" "{ft_project}"')

###RQ3 Text Gen cross dataset
def RQ3_TextGen_Cross():
    for llm in special_llms:
        for shot in shots:
            for project in projects:

                ft_projects = projects.copy()
                ft_projects.remove(project)

                for ft_project in ft_projects:
                    df_logs = pd.read_json(f"../logs/{project}_logs_validate_0.6.json")
                    for index, row in df_logs.iterrows():
                        file_index = row['index']
                        print(f'python textgen_prompt.py "{project}" "{file_index}" "validate" "../{llm}" "0.6" "{ft_project}"   ')

###RQ3 enlarged dataset
###RQ3 Fill Mask enlarged dataset
def RQ3_FillMask_Enlarged():
    for llm in pretrainedmodels:

        tokenizer = pretrained_to_model.get(llm)

        for shot in shots:
            for project in projects:
                df_logs = pd.read_json(f"../logs/{project}_logs_validate_0.6.json")
                for index, row in df_logs.iterrows():
                    file_index = row['index']
                    print(f'python fillmask_prompt_u.py "{project}" 8 "{tokenizer}" "../{llm}" "0.6"')


###RQ3 Text Gen enlarged dataset
def RQ3_TextGen_Enlarged():
    for llm in special_llms:
        for shot in shots:
            for project in projects:

                df_logs = pd.read_json(f"../logs/{project}_logs_validate_0.6.json")
                for index, row in df_logs.iterrows():
                    file_index = row['index']
                    print(f'python textgen_prompt.py "{project}" "{file_index}" "validate" "../{llm}" "0.6" "ultimate"   ')

### SETTINGS FOR EACH RQ AND CORRESPONDING METHOD
#RQ1
tokenizers = ["bert", "bert", "roberta", "roberta", "roberta", "roberta", "roberta", "roberta"]
pretrainedmodels = ["bert-base-uncased",
                    "bert-large-uncased",
                    "roberta-base",
                    "roberta-large",
                    "CodeBERTa-small-v1",
                    "codebert-base-mlm",
                    "codebert-java",
                    "graphcodebert-base"]
pretrained_to_model = {pretrained: model for pretrained, model in zip(pretrainedmodels, tokenizers)}
special_llms = ["Llama-2-7b-hf", "Llama-2-13b-hf", "CodeLlama-7b-hf", "CodeLlama-13b-hf"]
shots = ["0", "5", "0.5", "10", "20", "30", "0.6"]
RQ1_finetune()
RQ1_FillMask_results()
RQ1_TextGen_results()

#RQ2
tokenizer = ["roberta"]
pretrainedmodels = ["graphcodebert-base"]
pretrained_to_model = {pretrained: model for pretrained, model in zip(pretrainedmodels, tokenizer)}
special_llms = ["CodeLlama-7b-hf"]
shots = ["0", "5", "0.5", "10", "20", "30", "0.6"]
RQ2_finetune()
RQ2_FillMask_results()
RQ2_TextGen_results()

#RQ3
special_llms = ["CodeLlama-7b-hf"]
shots = ['0.6']
RQ3_finetune_enlarged()

tokenizer = ["roberta"]
pretrainedmodels = ["graphcodebert-base"]
pretrained_to_model = {pretrained: model for pretrained, model in zip(pretrainedmodels, tokenizer)}
special_llms = ["CodeLlama-7b-hf"]
shots = ["0.6"]
RQ3_FillMask_Cross()
RQ3_TextGen_Cross()
RQ3_FillMask_Enlarged()
RQ3_TextGen_Enlarged()

