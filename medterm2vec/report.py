
from .utils import loadingFiles

def jaccard_index(A, B):
    import numpy as np
    union = np.union1d(A, B).shape[0]
    if union==0: return 0.0
    else: return np.intersect1d(A, B).shape[0] / union

def statsInfo(logger, seq_data, demo_data, seq_len):
    logger.info("\n  <statsInfo>")
    if type(seq_data)!=list:
        seq_data = [seq_data]
    if type(demo_data)!=list:
        demo_data = [demo_data]
    if type(seq_len)!=list:
        seq_len = [seq_len]
        
    male_count = sum([p[0][0] for p in demo_data])
    avg_age_data = [sum([v[1] for v in p])/len(p) for p in demo_data]
    coCode_freq = [len(row.indices) for sprs_m, l in zip(seq_data, seq_len) for row in sprs_m[:l]]
    
    from scipy import stats
    stats_name = ['nobs', 'minmax', 'mean', 'variance', 'skewness', 'kurtosis']
    logger.info("\n  {0};".format('[gender level]'))
    for k, v in zip(['male', 'female', 'SUM'], [male_count, len(demo_data)-male_count, len(demo_data)]):
        logger.info("  {0:>12}: {1}".format(k, v))
    logger.info("\n  {0};".format('[avg_age info]'))
    for k, v in zip(stats_name, stats.describe(avg_age_data)):
        logger.info("  {0:>12}: {1}".format(k, v))
    logger.info("\n  {0} sequence length;".format('[visit level]'))
    for k, v in zip(stats_name, stats.describe(seq_len)):
        logger.info("  {0:>12}: {1}".format(k, v))
    logger.info("\n  {0} # of co-code;".format('[code level]'))
    for k, v in zip(stats_name, stats.describe(coCode_freq)):
        logger.info("  {0:>12}: {1}".format(k, v))
        
def topK_codeFrequency(logger, DATA_PATH, seq_data, topK):
    logger.info("\n  <topK_codeFrequency>\n")
    code2title = loadingFiles(logger, DATA_PATH, 'code2title.pkl')
    code2idx = loadingFiles(logger, DATA_PATH, 'code2idx.pkl')
    idx2code = {v:k for k,v in code2idx.items()}
    
    if type(seq_data)!=list:
        seq_data = [seq_data]
    from collections import Counter
    c = Counter([idx2code[idx] for sprs_m in seq_data for idx in sprs_m.indices]).most_common()[:topK]

    logger.info("\n  TopK of [{}] codes: ".format(len(c)))
    for idx, r in enumerate(c): 
        try: logger.info("  {0:>2}. [{1}]: {2}".format(idx, r, code2title[r[0]]))
        except: logger.info("  {0:>2}. [{1}]: {2}".format(idx, r, '???'))
            
def acc_codeFrequency(logger, DATA_PATH, seq_data, thr):
    logger.info("\n  <acc_codeFrequency>\n")
    code2title = loadingFiles(logger, DATA_PATH, 'code2title.pkl')
    code2idx = loadingFiles(logger, DATA_PATH, 'code2idx.pkl')
    idx2code = {v:k for k,v in code2idx.items()}
    
    if type(seq_data)!=list:
        seq_data = [seq_data]
    from collections import Counter
    c = Counter([idx2code[idx] for sprs_m in seq_data for idx in sprs_m.indices]).most_common()       
    logger.info("\n  Top accumulated freq ({}) of codes: ".format(thr))
    count_all = sum([freq for _, freq in c])
    count_acc = 0
    for idx, r in enumerate(c): 
        if count_acc/count_all >= thr: 
            logger.info("\tstop at {0}".format(count_acc/count_all))
            break
        try: logger.info("  {0:>2}. [{1}]: {2}".format(idx, r, code2title[r[0]]))
        except: logger.info("  {0:>2}. [{1}]: {2}".format(idx, r, '???'))
        count_acc += r[1]
        
def Get_datasets_info(RESULT_FOLDER_PATH, DATASETS, dataset_type, cohort_type, topK=15, thr=0.1):
    import logging, datetime, os
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(filename=os.path.join(RESULT_FOLDER_PATH, 'report_datasets_info.log'))
    logger.addHandler(file_handler)
    logger.info("\n{}".format(datetime.datetime.now()))
    
    logger.info("\n(REPORT) {}-{}".format(dataset_type, cohort_type))
    for ds_type, dataset in zip(['TRAIN', 'TEST'], [DATASETS.train, DATASETS.test]):
        if ds_type!=dataset_type: 
            if dataset_type!='ALL':
                continue
        #print('\n[{0}-{1}]'.format(ds_type, cohort_type))
        logger.info('\n[{0}-{1}]'.format(ds_type, cohort_type))
        if cohort_type=='TARGET':
            seq_data = dataset._t_ds._seq_data
            demo_data = dataset._t_ds._demo_data
            seq_len = dataset._t_ds._seq_len
        elif cohort_type=='COMP':
            seq_data = dataset._c_ds._seq_data
            demo_data = dataset._c_ds._demo_data
            seq_len = dataset._c_ds._seq_len
        elif cohort_type=='ALL':
            seq_data = dataset._t_ds._seq_data + dataset._c_ds._seq_data
            demo_data = dataset._t_ds._demo_data + dataset._c_ds._demo_data
            seq_len = dataset._t_ds._seq_len + dataset._c_ds._seq_len

        statsInfo(logger, seq_data, demo_data, seq_len)
        topK_codeFrequency(logger, DATASETS.info['DATA_FOLDER_PATH'], seq_data, topK)
        acc_codeFrequency(logger, DATASETS.info['DATA_FOLDER_PATH'], seq_data, thr)
    #print("\n  [DONE]\n\n")
    logger.info("\n  [DONE]\n\n")
    

def Stop_tensorboard():
    ## turn-off tensorboard service
    proc_list = get_ipython().getoutput(cmd='ps -ax | grep tensorboard')
    try:
        pid = [p.strip().split(' ')[0] for p in proc_list if 'grep' not in p][0]
        get_ipython().system_raw('kill -9 {}'.format(pid))
    
        import time
        time.sleep(1)
        print("[Stop] Tensorboard..\n")
    except:
        pass
    
def Run_tensorboard(LOGDIR):
    def _get_ip_address():
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]

    try:
        proc_list = get_ipython().getoutput(cmd='ps -ax | grep tensorboard')
    except:
        import subprocess
        proc_list = subprocess.check_output('ps -ax | grep tensorboard', 
                                            shell=True, universal_newlines=True).split('\n')

    IS_RUNNING = sum([1 if 'grep' not in p else 0 for p in proc_list]) > 0
    if IS_RUNNING:
        Stop_tensorboard()
    
    print("[Run] Tensorboard..\n")
    try:
        get_ipython().system_raw('tensorboard --logdir={} &'.format(LOGDIR))
    except:
        subprocess.call('tensorboard --logdir={} &'.format(LOGDIR), shell=True)
        
    import time
    time.sleep(1)
    try:
        IPADDRESS = _get_ip_address()
        if IPADDRESS.startswith('192') or IPADDRESS.startswith('10') or IPADDRESS.startswith('172'):
            IPADDRESS = 'localhost'
    except:
        IPADDRESS = 'localhost'
    print("[GO] http://{}:6006".format(IPADDRESS), "\n\n")
    

