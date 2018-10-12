
import numpy as np
import tensorflow as tf
from tqdm import trange
from . import dl_ops as ops

def param_dict_to_flag_grid(param_dict):
    flag_grid = dict()    
    for k, v in param_dict.items():
        if k in ['EMB_DATASETS', 'DATASETS_INFO', 'NEW_GAME', 'DATA_FOLDER_PATH', 'CONFIG_FOLDER_PATH', 'RESULT_FOLDER_PATH', 
                 'LEFT_CONTEXT_SIZE', 'RIGHT_CONTEXT_SIZE', 'DIRECTED', 'SKIP_EDGE_EXTRACTING',
                 'PROJECT_NAME', 'EMB_PARAMS_FILE_NAME', 'CDM_DB_NAME', ]:
            continue
        if k=='DUMPING_PATH':
            flag_grid[k] = [v]
        else:
            flag_grid[k] = v
    return flag_grid

def get_flag_list(flag_grid, FEATURE_SIZE):
    #update flag_grid 
    flag_grid['FEATURE_SIZE'] = [FEATURE_SIZE]
    
    #flag_grid to flag_list
    from itertools import product
    flag_list = [dict(list(zip(list(flag_grid.keys()), values))) 
                 for values in list(product(*flag_grid.values()))]
    
    #remove duplicated flag
    flag_list_new_unique = []
    for d_item in flag_list:
        if d_item not in flag_list_new_unique:
            flag_list_new_unique.append(d_item)
    return flag_list_new_unique

def get_model_list(param_dict):
    ##make flag_list
    flag_grid = param_dict_to_flag_grid(param_dict)
    flag_list = get_flag_list(flag_grid, param_dict['EMB_DATASETS'].info['FEATURE_SIZE'])
    
    ##get model_list
    MODEL_DICT = {'LINE_MODEL': LINE_MODEL}
    
    model_list = []
    for m_idx, flag in enumerate(flag_list):
        flag['MODEL_NAME'] = 'MODEL_{}'.format(m_idx+1)
        model_list.append(MODEL_DICT[flag['MODEL_ARCH']](flag))
    return model_list

class LINE_MODEL():
    def __init__(self, flag):
        self.flag = flag
        self.tensorDict = dict()
        self.resultDict = dict()
        self.g = tf.Graph()
        self.g_vis = tf.Graph()
        self.Building_graph()
        
    def _get_logger(self):
        from .utils import get_logger_instance
        self.logger = get_logger_instance(logger_name=self.flag['MODEL_NAME'], 
                                          DUMPING_PATH=self.flag['DUMPING_PATH'])
        
    def _basic_tensors(self):
        with tf.name_scope('Learning_Rate'):    
            self.tensorDict['global_step'] = tf.Variable(0, name="Global_step", trainable=False, dtype=tf.int32)
            if ('DECAY_STEPS' in self.flag.keys())&('DECAY_RATE' in self.flag.keys()):
                self.tensorDict['lr_p1'] = tf.train.exponential_decay(self.flag['LR_p1'], self.tensorDict['global_step'], 
                                                               self.flag['DECAY_STEPS'], self.flag['DECAY_RATE'], 
                                                               staircase=True, name='ExpDecay_lr_p1')
                self.tensorDict['lr_p2'] = tf.train.exponential_decay(self.flag['LR_p2'], self.tensorDict['global_step'], 
                                                               self.flag['DECAY_STEPS'], self.flag['DECAY_RATE'], 
                                                               staircase=True, name='ExpDecay_lr_p2')
            else:
                self.tensorDict['lr_p1'] = tf.constant(self.flag['LR_p1'], name='Constant_lr_p1')
                self.tensorDict['lr_p2'] = tf.constant(self.flag['LR_p2'], name='Constant_lr_p2')
                
    def _input_layer_tensors(self):
        with tf.name_scope('Input_Layer'):
            self.tensorDict['focus_w'] = tf.placeholder(tf.int32, shape=[self.flag['BATCH_SIZE']], name='focus_w')
            self.tensorDict['context_w'] = tf.placeholder(tf.int32, shape=[self.flag['BATCH_SIZE']], name='context_w')
            self.tensorDict['score'] = tf.placeholder(tf.float32, shape=[self.flag['BATCH_SIZE']], name='score')
            
    def _embedding_layer_tensors(self):
        with tf.name_scope('EMB_Layer'):
            ## p1
            self.tensorDict['p1_emb_matrix'] = tf.Variable(tf.random_uniform([self.flag['FEATURE_SIZE'], self.flag['EMB_SIZE']], 
                                                                             -0.01, 0.01), 
                                                           name='p1_emb_matrix')
            self.tensorDict['p1_focus_emb_vec'] = tf.nn.embedding_lookup(self.tensorDict['p1_emb_matrix'],
                                                                         self.tensorDict['focus_w'], 
                                                                         name='p1_focus_emb_vec')
            self.tensorDict['p1_context_emb_vec'] = tf.nn.embedding_lookup(self.tensorDict['p1_emb_matrix'],
                                                                           self.tensorDict['context_w'], 
                                                                           name='p1_context_emb_vec')
            
            ## p2
            self.tensorDict['p2_emb_focus_matrix'] = tf.Variable(tf.random_uniform([self.flag['FEATURE_SIZE'], 
                                                                                    self.flag['EMB_SIZE']], -0.01, 0.01), 
                                                                 name='p2_emb_focus_matrix')
            self.tensorDict['p2_focus_emb_vec'] = tf.nn.embedding_lookup(self.tensorDict['p2_emb_focus_matrix'], 
                                                                         self.tensorDict['focus_w'],
                                                                         name='p2_focus_emb_vec')
            
            self.tensorDict['p2_emb_context_matrix'] = tf.Variable(tf.random_uniform([self.flag['FEATURE_SIZE'], 
                                                                                      self.flag['EMB_SIZE']], 
                                                                             -0.01, 0.01), 
                                                                   name='p2_emb_context_matrix')
            self.tensorDict['p2_context_emb_vec'] = tf.nn.embedding_lookup(self.tensorDict['p2_emb_context_matrix'],
                                                                           self.tensorDict['context_w'],
                                                                           name='p2_context_emb_vec')
    
    def _Inference(self):        
        with tf.name_scope('Inference'):
            self._embedding_layer_tensors()
        
    def _Loss(self):
        with tf.variable_scope('Loss'):
            with tf.name_scope('First-order_proximity'):
                p1_inner_product = tf.reduce_sum((self.tensorDict['p1_focus_emb_vec'] * self.tensorDict['p1_context_emb_vec']), 
                                                 axis=1, name='p1_inner_product')
                p1_logits = tf.nn.sigmoid(p1_inner_product, name='p1_logits')
                self.tensorDict['loss_1st_prox'] = tf.reduce_mean(-self.tensorDict['score']*tf.log(p1_logits+1e-10), 
                                                                  name='loss_1st_prox')

            with tf.name_scope('Second-order_proximity'):
                p2_inner_product = tf.reduce_sum((self.tensorDict['p2_focus_emb_vec'] * self.tensorDict['p2_context_emb_vec']), 
                                                 axis=1, name='p2_inner_product')
                #issue; add negative sampling
                p2_numerator = tf.exp(p2_inner_product, name='p2_numerator')
                p2_denominator = tf.reduce_sum(tf.exp(tf.matmul(self.tensorDict['p2_focus_emb_vec'], 
                                                                tf.transpose(self.tensorDict['p2_emb_context_matrix']))) + 1e-10, 
                                               axis=1, name='p2_denominator') 
                p2_logits = tf.divide(p2_numerator, p2_denominator, name='p2_logits')
                self.tensorDict['loss_2nd_prox'] = tf.reduce_mean(-self.tensorDict['score']*tf.log(p2_logits+1e-10), 
                                                                  name='loss_2nd_prox')
                        
    def _Optimizer(self):
        with tf.name_scope('Optimizer_p1'):       
            optimizer_p1 = tf.train.AdamOptimizer(self.tensorDict['lr_p1'], name='optimizer_p1')
            self.tensorDict['trainOp_p1'] = optimizer_p1.minimize(self.tensorDict['loss_1st_prox'], 
                                                                  global_step=self.tensorDict['global_step'])
        with tf.name_scope('Optimizer_p2'):       
            optimizer_p2= tf.train.AdamOptimizer(self.tensorDict['lr_p2'], name='optimizer_p2')
            self.tensorDict['trainOp_p2'] = optimizer_p2.minimize(self.tensorDict['loss_2nd_prox'], 
                                                                  global_step=self.tensorDict['global_step'])
            
    def _Summary(self):
        ## logging
        self.logger.info("\n[FLAG]")
        for k, v in self.flag.items():
            self.logger.info("\t{}:  {}".format(k, v))
            
        self.logger.info("\n[INPUT_LAYERS]")
        self.logger.info("\tfocus_w: {}".format(self.tensorDict['focus_w']))
        self.logger.info("\tcontext_w: {}".format(self.tensorDict['context_w']))
        self.logger.info("\tscore: {}".format(self.tensorDict['score']))
        
        self.logger.info("\n[EMB_LAYERS]")
        self.logger.info("\tp1_emb_matrix: {}".format(self.tensorDict['p1_emb_matrix']))
        self.logger.info("\tp1_focus_emb_vec: {}".format(self.tensorDict['p1_focus_emb_vec']))
        self.logger.info("\tp1_context_emb_vec: {}".format(self.tensorDict['p1_context_emb_vec']))
        self.logger.info("\tp2_emb_focus_matrix: {}".format(self.tensorDict['p2_emb_focus_matrix']))
        self.logger.info("\tp2_focus_emb_vec: {}".format(self.tensorDict['p2_focus_emb_vec']))
        self.logger.info("\tp2_emb_context_matrix: {}".format(self.tensorDict['p2_emb_context_matrix']))
        self.logger.info("\tp2_context_emb_vec: {}".format(self.tensorDict['p2_context_emb_vec']))
                    
        self.logger.info("\n[LOSS]")
        self.logger.info("\tloss_1st_prox: {}".format(self.tensorDict['loss_1st_prox']))
        self.logger.info("\tloss_2nd_prox: {}".format(self.tensorDict['loss_2nd_prox']))
        
        ## summary
        tf.summary.scalar('lr_p1', self.tensorDict['lr_p1'])
        tf.summary.scalar('lr_p2', self.tensorDict['lr_p2'])
        tf.summary.scalar('loss_1st_prox', self.tensorDict['loss_1st_prox'])
        tf.summary.scalar('loss_2nd_prox', self.tensorDict['loss_2nd_prox'])
        
    def Building_graph(self):
        with self.g.as_default():
            self._basic_tensors()
            self._input_layer_tensors()
            self._Inference()
            self._Loss()
            self._Optimizer()
            self._get_logger()
            self._Summary()
            
            

