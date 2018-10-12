
def Run(**kwargs): 
    """
    [Essential]
      "Load params from 'CONFIG_FOLDER_PATH/EMB_PARAMS_FILE_NAME'"
    CONFIG_FOLDER_PATH; (e.g.) '/path/to/CONFIG'
    DATA_FOLDER_PATH; (e.g.) '/path/to/DATA/'
    RESULT_FOLDER_PATH; (e.g.) '/path/to/RESULT'
    PROJECT_NAME; (e.g.) 'project'
    EMB_PARAMS_FILE_NAME; (e.g.) 'EMB_PARAMS.txt'
    DATASETS_INFO; dataset.info
    SKIP_EDGE_EXTRACTING; (e.g.) True
    NEW_GAME; (e.g) False 
    
    ######################### ALL PARAMS ##############################
    
    [basics]
    CONFIG_FOLDER_PATH; (e.g.) '/path/to/CONFIG'
    DATA_FOLDER_PATH; (e.g.) '/path/to/DATA/'
    RESULT_FOLDER_PATH; (e.g.) '/path/to/RESULT'
    PROJECT_NAME; (e.g.) 'project'
    CDM_DB_NAME; (e.g.) 'cdm_db_name'
    EMB_PARAMS_FILE_NAME; (e.g.) 'EMB_PARAMS.txt'
    
    [runtime_params]
    DATASETS; a dataset object
    SKIP_EDGE_EXTRACTING; (e.g.) True
    NEW_GAME; (e.g) False 
    
    [model_params]
    # about edge_extracting
    LEFT_CONTEXT_SIZE; (e.g.) 2
    RIGHT_CONTEXT_SIZE; (e.g.) 2
    DIRECTED; (e.g.) False

    MODEL_ARCH; (e.g.) ['LINE_MODEL']
    BATCH_SIZE; (e.g.) [128]
    EMB_SIZE; (e.g.) [32, 64]
    
    # about emb_model
    MODEL_ARCH; (e.g.) [LINE_MODEL]
    BATCH_SIZE; (e.g.) [128]
    EMB_SIZE; (e.g.) [32, 64]
    LR_p1; (e.g.) [5e-1, 5e-2]
    LR_p2; (e.g.) [5e-2]
    DECAY_STEPS; (e.g.) [1000]
    DECAY_RATE; (e.g.) [0.9]
    TRAIN_STEPS_p1; (e.g.) [1000]
    TRAIN_STEPS_p2; (e.g.) [1000]
    PRINT_BY; (e.g.) [2000]
    """
    from .utils import get_logger_instance, option_printer, dumpingFiles, loadingFiles, get_param_dict
    from .model import get_model_list
    from .train import Train_model_list
    from .report import Get_datasets_info
    from .emb_dataset import Get_emb_dataset
    import os, glob
    import logging, datetime
    from importlib import reload
    
    ## get params
    param_dict = get_param_dict(kwargs['EMB_PARAMS_FILE_NAME'], kwargs['CONFIG_FOLDER_PATH'])
    param_dict.update(kwargs)
    
    param_dict['DUMPING_PATH'] = os.path.join(param_dict['RESULT_FOLDER_PATH'], 
                                              param_dict['PROJECT_NAME'], 
                                              param_dict['DATASETS_INFO']['CDM_DB_NAME'])
        
    if param_dict['NEW_GAME']:
        print("[!!] Are you sure NEW_GAME is True?; \n\t(REMOVE ALL RESULTS AND START OVER)")
        confirm = input()
        if confirm.lower() in ['y', 'yes', 'true']:
            print("\n\t(NEW_GAME => True)")
            import shutil, glob, os
            _ = [shutil.rmtree(p) for p in glob.glob(param_dict['DUMPING_PATH'])] #remove param_dict['DUMPING_PATH']
        else:
            print("\n\t(NEW_GAME => False)")
            param_dict['NEW_GAME'] = False   

    if not os.path.exists(param_dict['DUMPING_PATH']): 
        os.makedirs(param_dict['DUMPING_PATH'])
    
    ## logger
    logging.shutdown()
    reload(logging)
    main_logger = get_logger_instance(logger_name='emb_pipeline', 
                                      DUMPING_PATH=param_dict['DUMPING_PATH'], 
                                      parent_name=False,
                                      stream=True)
    
    ## print options
    main_logger.info("\n (params) \n")
    option_printer(main_logger, **param_dict)
    main_logger.info("="*100 + "\n")
    
    ## [1] Make datasets for emb
    param_dict['EMB_DATASETS'] = Get_emb_dataset(**param_dict)
    param_dict['EMB_DATASETS'].info['PROJECT_NAME'] = param_dict['PROJECT_NAME']
    param_dict['EMB_DATASETS'].info['CDM_DB_NAME'] = param_dict['DATASETS_INFO']['CDM_DB_NAME']
    
    main_logger.info("\n[Emb_dataset Info.]\n")
    option_printer(main_logger, **param_dict['EMB_DATASETS'].info)
    main_logger.info("="*100 + "\n")
    
    ## [2] Make Models
    main_logger.info("\n[EMB_model Setting]\n")
    model_list = get_model_list(param_dict)
    main_logger.info("="*100 + "\n")
    
    ## [3] Train Models
    Train_model_list(MODEL_LIST = model_list,
                     DATASETS = param_dict['EMB_DATASETS'],
                     DUMPING_PATH = param_dict['DUMPING_PATH'], 
                     new_game = param_dict['NEW_GAME'])
    main_logger.info("="*100 + "\n")
    
    ## [4] Get Results
    main_logger.info("\n[Model_results]\n")
    df_emb_results = loadingFiles(main_logger, param_dict['DUMPING_PATH'], 'df_emb_RESULTS.pkl')
    
    main_logger.info("\nALL DONE!!")
    for h in list(main_logger.handlers):
        main_logger.removeHandler(h)
        h.flush()
        h.close()
    return df_emb_results
    