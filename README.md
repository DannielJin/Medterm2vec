# medterm2vec
Extracting edge information and making embedding matrix from the medical sequence data
1. Extracting edge information
2. Making datasets for embedding
3. Supports graph embedding methods (e.g.) LINE, SDNE
4. Monitoring by tensorboard

## Dependency

Install cdm_datasetmaker:
```sh
git clone https://github.com/DannielJin/cdm_datasetmaker.git
pip3 install cdm_datasetmaker/dist/*.whl
```

## Installation

At /PATH/TO/THE/PACKAGE/FOLDER/:

```sh
pip3 install ./dist/*.whl
```
```sh
pip3 uninstall ./dist/*.whl -y
```

## Usage example

* SAMPLE_CODES in the EXAMPLE folder    

Guidelines:  
(1) In your project folder, (e.g.) /PRJ/  
(2) Make folders;   
    - CONFIG  
    - DATA  
    - RESULT    
(3) Run the codes in your project folder. 
  
  
Make EMB_PARAMS.txt in CONFIG FOLDER:
```
## (e.g.)
# About edge_extracting
LEFT_CONTEXT_SIZE = 2      #LEFT_CONTEXT_SIZE to get 'source-target' scores
RIGHT_CONTEXT_SIZE = 2     #RIGHT_CONTEXT_SIZE to get 'source-target' scores
DIRECTED = False           #If False, 'source-target-value' == 'target-source-value'

# About emb_model (multiple items for grid search)
MODEL_ARCH = LINE_MODEL    #LINE_MODEL
BATCH_SIZE = 128
EMB_SIZE = 32, 64          #Embedding size
LR_p1 = 5e-2
LR_p2 = 1e-5, 5e-5
DECAY_STEPS = 1000
DECAY_RATE = 0.9
TRAIN_STEPS_p1 = 10000     #train for 1st-prox (edge info)
TRAIN_STEPS_p2 = 10000     #train for 2nd-prox (neighbor info)
PRINT_BY = 2000
```

Main codes:
1. get datasets by cdm_datasetmaker:
```
from cdm_datasetmaker import Get_datasets
datasets = Get_datasets(CONFIG_FOLDER_PATH = '/PRJ/CONFIG/',       #/PATH/TO/CONFIG/FOLDER/
                        DATA_FOLDER_PATH = '/PRJ/DATA/',           #/PATH/TO/DATA/FOLDER/ (save&load)
                        RESULT_FOLDER_PATH = '/PRJ/RESULT/',       #/PATH/TO/RESULT/FOLDER/ (logs)
                        PROJECT_NAME = 'PROJECT_DATASETS',         #PROJECT_NAMES
                        DB_CONN_FILENAME = 'DB_connection.txt',
                        DS_PARAMS_FILE_NAME = 'DS_PARAMS.txt', 
                        PIPELINE_START_LEVEL = 4)                  #Starting level
```
PIPELINE_START_LEVEL; 
    1. Make_target_comp_tables  (when the first time)
    2. Table to rawSeq
    3. RawSeq to multihot
    4. Multihot to Dataset      (when you want to restore datasets)

2. Run pipeline:
```
from medterm2vec import Run
df_emb_results = Run(CONFIG_FOLDER_PATH = '/PRJ/CONFIG/',          #/PATH/TO/CONFIG/FOLDER/
                     DATA_FOLDER_PATH = '/PRJ/DATA/',              #/PATH/TO/DATA/FOLDER/ (cdm_datasetmaker)
                     RESULT_FOLDER_PATH = '/PRJ/RESULT/',          #/PATH/TO/RESULT/FOLDER/ (logs, model save&load)
                     PROJECT_NAME = 'PROJECT_EMB',                 #PROJECT_NAMES
                     EMB_PARAMS_FILE_NAME = 'EMB_PARAMS.txt', 
                     DATASETS_INFO = datasets.info,                #datasets.info from the datasets object
                     SKIP_EDGE_EXTRACTING = True,                  #If False, edge_extracting process will be started
                     NEW_GAME = True)                              #Fresh starting for training embedding models 
```
If SKIP_EDGE_EXTRACTING==True, edge_extracting process will be started.  
Else, dumped 'df_edge.pkl' will be used.


When you run the pipeline, tensorboard service will be started. 
If you run the codes in jupyter notebook, click the tensorboard_service_address (printed)
Or, you can run tensorboard manually.
```
from medterm2vec.report import Run_tensorboard
Run_tensorboard('/RESULT_FOLDER_PATH/PROJECT_NAME/CDM_DB_NAME/')
```
```
from medterm2vec.report import Stop_tensorboard
Stop_tensorboard()
```

## Release History

* 1.0.0
    * released

## Meta

Sanghyung Jin, MS(1) – jsh90612@gmail.com  
Yourim Lee, BS(1) - urimeeee.e@gmail.com  
Rae Woong Park, MD, PhD(1)(2) - rwpark99@gmail.com  

(1) Dept. of Biomedical Informatics, Ajou University School of Medicine, Suwon, South Korea  
(2) Dept. of Biomedical Sciences, Ajou University Graduate School of Medicine, Suwon, South Korea  

