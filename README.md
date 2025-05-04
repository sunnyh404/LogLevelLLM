# Benchmarking Open-Source Large Language Models for Log Level Suggestion

This repo is the package for "Benchmarking Open-Source Large Language Models for Log Level Suggestion".

## Study Design

<p align="center"><img src="docs/overview_diagram.png" width="1000"></p>



## Structure
We present the repository structure below.

```
.
├── README.md
├── data
│   ├── data-HBase.json.zip
│   ├── data-JMeter.json.zip
│   ├── data-Zookeeper.json.zip
│   ├── data-cassandra.json.zip
│   ├── data-elasticsearch.json.zip
│   ├── data-flink.json.zip
│   ├── data-kafka.json.zip
│   ├── data-karaf.json.zip
│   ├── data-wicket.json.zip
│   └── README.md
├── docs
│   └── overview_diagram.pdf
├── evaluation
│   ├── RQ1
│   │   └── RQ1.png
│   ├── RQ2
│   │   └── RQ2.png
│   └── RQ3
│       └── RQ3.png
├── models
│   └── README.md
├── README.md
├── requirements.txt
├── RQ1_results.py
├── RQ2_results.py
├── RQ3_results.py
├── calculations_util.py
├── directory.txt
├── fillmask_prompt.py
├── fillmask_prompt_rq2.py
├── fillmask_prompt_u.py
├── fillmask_prompt_x.py
├── generate_command.py
├── prompter.py
├── sampling.py
└── textgen_prompt_rq2.py
```


## Requirement 

```shell
pip install -r requirements.txt
```

## Models download

Please download the base LLMs from links in [GitHub](https://github.com/LogLevelLLM/LogLevelLLM/tree/main/models).


## Datasets download

Please first download the full datasets from links in [GitHub](https://github.com/LogLevelLLM/LogLevelLLM/tree/main/data).


## Instructions:
1. Run directory.txt to create the folders.
2. Run sampling.py to prepare the data.
3. Run generate_command.py to generate the python commands needed to finetune and generate results.
4. Run RQ1_results.py, RQ2_results.py, RQ3_results.py to process the results. 


## Evaluation Results
### RQ1: What is the accuracy of LLMs in suggesting log level? 
<p align="center"><img src="evaluation/RQ1/RQ1.png" width="800"></p>

### RQ2: How effective is including additional context help in log level suggestion?
<p align="center"><img src="evaluation/RQ2/RQ2.png" width="800"></p>

### RQ3: How is the generalizability of LLMs in suggesting log levels?
<p align="center"><img src="evaluation/RQ3/RQ3.png" width="500"></p>
