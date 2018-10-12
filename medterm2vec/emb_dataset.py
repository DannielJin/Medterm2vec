import numpy as np
from .utils import dumpingFiles, loadingFiles, option_printer

def edge_extractor(logger, seq_data, seq_len, left_context_size, right_context_size, directed):
    import pandas as pd
    from collections import defaultdict
    edge_dict = defaultdict(float)
    
    data_len = len(seq_len)
    printBy = int(data_len/10)
    for i, (sprs_m, l) in enumerate(zip(seq_data, seq_len)):
        if (((i+1)%printBy)==0) or i==0: 
            logger.info("  ..({}/{})".format(i+1, data_len))
        vseq = sprs_m[:l]
        source_size = vseq.shape[0]
        for s_idx in range(source_size):
            source_list = vseq[s_idx].indices
            # in visit
            for s1 in source_list:
                for s2 in source_list:
                    discounted_value = 1
                    if s1 != s2:
                        edge_dict[(s1, s2)] += discounted_value
            # between visits
            target = vseq[s_idx]
            left_context_list = vseq[max(0,s_idx-left_context_size) : s_idx]
            right_context_list = vseq[s_idx+1 : min((s_idx+right_context_size)+1, source_size)]
            #left
            for idx, contexts in enumerate(left_context_list[::-1]):
                discounted_value = 1/(idx+1)
                for t in target.indices:
                    for c in contexts.indices:
                        edge_dict[(t, c)] += discounted_value
            #right
            for idx, contexts in enumerate(right_context_list):
                discounted_value = 1/(idx+1)
                for t in target.indices:
                    for c in contexts.indices:
                        edge_dict[(t, c)] += discounted_value

    if not directed:
        new_dict = dict()
        for (t, c), v in edge_dict.items():
            if (c, t) in new_dict.keys():
                new_dict[(c, t)] += v
            else:
                new_dict[(t, c)] = v
        edge_dict = new_dict
    
    df_edge = pd.DataFrame([[s, t, v] for (s, t), v in edge_dict.items()], 
                           columns=['source', 'target', 'value']).astype({'source':'int', 'target':'int'})
    return df_edge

class Graph_DataSet():
    def __init__(self, df, code2idx):
        self._num_examples = len(df)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.code2idx = code2idx
        self.targets = df.iloc[:,0].values
        self.contexts = df.iloc[:,1].values
        self.scaled_scores = df.iloc[:,2].values
         
    def _shuffle(self, targets, contexts, scaled_scores):
        import sklearn as sk
        return sk.utils.shuffle(targets, contexts, scaled_scores)
    
    def get_adj_matrix(self):
        import numpy as np
        adj_matrix = np.zeros([len(self.code2idx), len(self.code2idx)])
        for t, c, v in zip(self.targets, self.contexts, self.scaled_scores):
            adj_matrix[t, c] = v
            adj_matrix[c, t] = v
        return adj_matrix

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        end = self._index_in_epoch
        if end<=self._num_examples:
            return self.targets[start:end], self.contexts[start:end], self.scaled_scores[start:end]
        else:
            self._epochs_completed += 1
            num_of_short = batch_size-(self._num_examples-start)
            num_of_extra_batch = num_of_short // self._num_examples
            num_of_extra_example = num_of_short % self._num_examples
            self._epochs_completed += num_of_extra_batch
            self._index_in_epoch = num_of_extra_example

            tmp_targets=self.targets[start:]; tmp_contexts=self.contexts[start:]; tmp_scaled_scores=self.scaled_scores[start:]      
            self.targets, self.contexts, self.scaled_scores = self._shuffle(self.targets, self.contexts, self.scaled_scores)
            batch_targets = np.concatenate([tmp_targets]+[self.targets]*num_of_extra_batch
                                           +[self.targets[0:num_of_extra_example]], axis=0)
            batch_contexts = np.concatenate([tmp_contexts]+[self.contexts]*num_of_extra_batch
                                            +[self.contexts[0:num_of_extra_example]], axis=0)
            batch_scaled_scores = np.concatenate([tmp_scaled_scores]+[self.scaled_scores]*num_of_extra_batch
                                                 +[self.scaled_scores[0:num_of_extra_example]], axis=0)
            return batch_targets, batch_contexts, batch_scaled_scores 

def Get_emb_dataset(**kwargs):
    import os, datetime
    from .utils import get_logger_instance
    
    if not os.path.exists(kwargs['DUMPING_PATH']): 
        os.makedirs(kwargs['DUMPING_PATH'])
    
    logger = get_logger_instance(logger_name='dataset', 
                                 DUMPING_PATH=kwargs['DUMPING_PATH'], 
                                 parent_name='emb_pipeline', 
                                 stream=False)
    logger.info("\n{}".format(datetime.datetime.now()))
    logger.info("[Get_emb_dataset]")
    
    #extracting df_edge
    if kwargs['SKIP_EDGE_EXTRACTING']:
        logger.info("\n  (skip edge_extracting)")
        df_edge = loadingFiles(logger, kwargs['DATA_FOLDER_PATH'], 'df_edge.pkl')
    else:
        logger.info("\n  (extracting df_edge)")
        seq_data = loadingFiles(logger, kwargs['DATA_FOLDER_PATH'], 't_seq_data.pkl') + loadingFiles(logger, kwargs['DATA_FOLDER_PATH'], 'c_seq_data.pkl')
        seq_len = loadingFiles(logger, kwargs['DATA_FOLDER_PATH'], 't_seq_len.pkl') + loadingFiles(logger, kwargs['DATA_FOLDER_PATH'], 'c_seq_len.pkl')
        df_edge = edge_extractor(logger, seq_data, seq_len, kwargs['LEFT_CONTEXT_SIZE'][0], kwargs['RIGHT_CONTEXT_SIZE'][0], kwargs['DIRECTED'])
        #ceiling and scaling.
        dumpingFiles(logger, kwargs['DATA_FOLDER_PATH'], 'df_edge.pkl', df_edge)
    
    #make dataset
    logger.info("\n  (make emb_dataset)")
    code2idx = loadingFiles(logger, kwargs['DATA_FOLDER_PATH'], 'code2idx.pkl')
    dataset = Graph_DataSet(df_edge, code2idx)
    dataset.info = dict()
    dataset.info['DATA_FOLDER_PATH'] = kwargs['DATA_FOLDER_PATH']
    dataset.info['RESULT_FOLDER_PATH'] = kwargs['RESULT_FOLDER_PATH']
    dataset.info['LEFT_CONTEXT_SIZE'] = kwargs['LEFT_CONTEXT_SIZE']
    dataset.info['RIGHT_CONTEXT_SIZE'] = kwargs['RIGHT_CONTEXT_SIZE']
    dataset.info['DIRECTED'] = kwargs['DIRECTED']
    dataset.info['FEATURE_SIZE'] = len(code2idx)
    return dataset
    
    