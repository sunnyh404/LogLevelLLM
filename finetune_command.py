

from sys import exit
#
#
#
# string = """
# echo "wicket 5 CodeLlama-7b-hf"
# time python finetune_codellama.py         --base_model "../CodeLlama-7b-hf"         --data_path "logs/wicket_instructions_logs_train_5.json"         --output_dir "saved_model/CodeLlama-7b-hf/wicket_0.5"         --batch_size 2         --micro_batch_size 2         --num_epochs 10         --learning_rate 5e-4         --cutoff_len 2048         --val_set_size 0         --lora_r 8         --lora_alpha 16         --lora_dropout 0.05         --lora_target_modules '[q_proj,v_proj]'         --train_on_inputs         --group_by_length > "saved_model/CodeLlama-7b-hf/wicket_0.5_finetune.txt"
#
# echo "wicket 10 CodeLlama-7b-hf"
# time python finetune_codellama.py         --base_model "../CodeLlama-7b-hf"         --data_path "logs/wicket_instructions_logs_train_10.json"         --output_dir "saved_model/CodeLlama-7b-hf/wicket_10"         --batch_size 2         --micro_batch_size 2         --num_epochs 10         --learning_rate 5e-4         --cutoff_len 2048         --val_set_size 0         --lora_r 8         --lora_alpha 16         --lora_dropout 0.05         --lora_target_modules '[q_proj,v_proj]'         --train_on_inputs         --group_by_length > "saved_model/CodeLlama-7b-hf/wicket_10_finetune.txt"
#
# echo "wicket 20 CodeLlama-7b-hf"
# time python finetune_codellama.py         --base_model "../CodeLlama-7b-hf"         --data_path "logs/wicket_instructions_logs_train_20.json"         --output_dir "saved_model/CodeLlama-7b-hf/wicket_20"         --batch_size 2         --micro_batch_size 2         --num_epochs 10         --learning_rate 5e-4         --cutoff_len 2048         --val_set_size 0         --lora_r 8         --lora_alpha 16         --lora_dropout 0.05         --lora_target_modules '[q_proj,v_proj]'         --train_on_inputs         --group_by_length > "saved_model/CodeLlama-7b-hf/wicket_20_finetune.txt"
#
# echo "wicket 30 CodeLlama-7b-hf"
# time python finetune_codellama.py         --base_model "../CodeLlama-7b-hf"         --data_path "logs/wicket_instructions_logs_train_30.json"         --output_dir "saved_model/CodeLlama-7b-hf/wicket_30"         --batch_size 2         --micro_batch_size 2         --num_epochs 10         --learning_rate 5e-4         --cutoff_len 2048         --val_set_size 0         --lora_r 8         --lora_alpha 16         --lora_dropout 0.05         --lora_target_modules '[q_proj,v_proj]'         --train_on_inputs         --group_by_length > "saved_model/CodeLlama-7b-hf/wicket_30_finetune.txt"
#
# # echo "wicket 0.6 CodeLlama-7b-hf"
# # time python finetune_codellama.py         --base_model "../CodeLlama-7b-hf"         --data_path "logs/wicket_instructions_logs_train_0.6.json"         --output_dir "saved_model/CodeLlama-7b-hf/wicket_0.6"         --batch_size 2         --micro_batch_size 2         --num_epochs 10         --learning_rate 5e-4         --cutoff_len 2048         --val_set_size 0         --lora_r 8         --lora_alpha 16         --lora_dropout 0.05         --lora_target_modules '[q_proj,v_proj]'         --train_on_inputs         --group_by_length > "saved_model/CodeLlama-7b-hf/wicket_0.6_finetune.txt"
#
# """
# string = string.replace("wicket", "HBase")
# print(string)
# string = string.replace("python finetune_codellama.py", "python finetune_llama2.py")
# # print(string.replace("CodeLlama-7b-hf", "CodeLlama-13b-hf"))
# print(string.replace("CodeLlama-7b-hf", "Llama-2-13b-hf"))


PROJECTS = ["elasticsearch", "flink", "HBase", "JMeter", "kafka", "karaf", "wicket", "Zookeeper"]


for project in PROJECTS:
    filename = f"bash/hp_{project}.sh"
    print(f"sbatch {filename}")
exit()

SHOTS = ["10", "20", "30", "0.6"]

MODELS = ["bert", "bert", "roberta", "roberta", "roberta", "roberta", "roberta", "roberta"]
LLMS = ["bert-base-uncased", "bert-large-uncased", "roberta-base", "roberta-large", "CodeBERTa-small-v1", "codebert-base-mlm", "codebert-java", "graphcodebert-base"]
pretrained_to_model = {pretrained: model for pretrained, model in zip(LLMS, MODELS)}

string = """

echo "{project} {shot} {llm}"
python hard_prompt.py "{project}" 8 "{model}" "/home/he_yiwen/prompt/{llm}" "{shot}"

"""


for project in PROJECTS:
    filename = f"hp_{project}.sh"
    
    with open(f'bash/{filename}', 'w') as f:
        f.write("#!/bin/bash")
        f.write("\n")
        f.write(f"#SBATCH -J hp_{project} --mem=120GB --gpus=2 --account=he_yiwen ")
        f.write("\n")
        f.write("#SBATCH --mail-type=BEGIN,END ")
        f.write("\n")
        f.write("#SBATCH --mail-user=yiwen_90@hotmail.com")

        f.write("\n")
        f.write("\n")
        f.write("source /etc/profile.d/modules.sh")
        f.write("\n")
        f.write("module load anaconda/3.2023.09")
        f.write("\n")
        f.write("module load cuda/12.2.2")
        f.write("\n")
        f.write("source activate py39")
        f.write("\n\n")
        f.write(f'echo "{filename}"')

    for shot in SHOTS:

        for llm in LLMS:
            
            model = pretrained_to_model.get(llm)
                
            command = string.replace("{project}", project)
            command = command.replace("{shot}", shot)
            command = command.replace("{model}", model)
            command = command.replace("{llm}", llm)
            with open(f'bash/{filename}', 'a') as f:
                f.write(command)