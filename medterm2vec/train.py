
import tensorflow as tf
from tqdm import trange
import numpy as np
from .utils import get_logger_instance, dumpingFiles, option_printer

def Train_model(logger, dataSets, model, logPath, new_game=True):
    import os
    
    with tf.Session(graph=model.g) as sess:
        tf.global_variables_initializer().run(session=sess)
        tf.train.start_queue_runners(sess=sess)
        tr_writer = tf.summary.FileWriter(logPath+'/train', graph=sess.graph)
        summary_op = tf.summary.merge_all()
        
        ## LOAD
        if not new_game: loading = load_model(logger, logPath, sess)
        else: 
            if logger is not None:
                logger.info("[@] New game..")
            else:
                print("[@] New game..")
        
        if logger is not None: 
            logger.info("\ntraining p1..\n")
        else:
            print("\ntraining p1..\n")
        for i in trange(model.flag['TRAIN_STEPS_p1']):
            batch_data = dataSets.next_batch(model.flag['BATCH_SIZE'])
            feed_dict = {model.tensorDict['focus_w']: batch_data[0], 
                         model.tensorDict['context_w']: batch_data[1], 
                         model.tensorDict['score']: batch_data[2]}
            _ = sess.run(model.tensorDict['trainOp_p1'], feed_dict=feed_dict)
            
            if (i%model.flag['PRINT_BY'])==0:
                g_step, lr_p1, loss_p1 = sess.run([model.tensorDict['global_step'], model.tensorDict['lr_p1'], 
                                                   model.tensorDict['loss_1st_prox']], 
                                                  feed_dict=feed_dict)
                tr_summary = sess.run(summary_op, feed_dict=feed_dict)
                tr_writer.add_summary(tr_summary, g_step)
                if logger is not None:
                    logger.info('[G-{}]  LOSS ({:.4f}) LR_p1 ({:6f})'.format(g_step, loss_p1, lr_p1))
                else:
                    print('[G-{}]  LOSS ({:.4f})  LR_p1 ({:6f})'.format(g_step, loss_p1, lr_p1))
                    
        ## SAVE
        save_model(logger, logPath, sess, g_step=model.tensorDict['global_step'])
        
        
        if logger is not None: 
            logger.info("\ntraining p2..\n")
        else:
            print("\ntraining p2..\n")
        for i in trange(model.flag['TRAIN_STEPS_p2']):
            batch_data = dataSets.next_batch(model.flag['BATCH_SIZE'])
            feed_dict = {model.tensorDict['focus_w']: batch_data[0], 
                         model.tensorDict['context_w']: batch_data[1], 
                         model.tensorDict['score']: batch_data[2]}
            _ = sess.run(model.tensorDict['trainOp_p2'], feed_dict=feed_dict)
            
            if (i%model.flag['PRINT_BY'])==0:
                g_step, lr_p2, loss_p1, loss_p2 = sess.run([model.tensorDict['global_step'], model.tensorDict['lr_p2'], 
                                                         model.tensorDict['loss_1st_prox'], model.tensorDict['loss_2nd_prox']], 
                                                           feed_dict=feed_dict)
                tr_summary = sess.run(summary_op, feed_dict=feed_dict)
                tr_writer.add_summary(tr_summary, g_step)
                if logger is not None:
                    logger.info('[G-{}]  LOSS ({:.4f} / {:.4f}) LR_p2 ({:6f})'.format(g_step, loss_p1, loss_p2, lr_p2))
                else:
                    print('[G-{}]  LOSS ({:.4f} / {:.4f}) LR_p2 ({:6f})'.format(g_step, loss_p1, loss_p2, lr_p2))
                
        ## SAVE
        save_model(logger, logPath, sess, g_step=model.tensorDict['global_step'])

        if logger is not None: 
            logger.info("\n[Training DONE]")
        else:
            print("\n[Training DONE]")
        
        
        ## dumping embedding_matrix
        if logger is not None:
            logger.info("\n[Dumping emb_matrix]..")
        else:
            print("\n[Dumping emb_matrix]..")
        p1_emb_matrix = sess.run(model.tensorDict['p1_emb_matrix'])
        p2_emb_focus_matrix = sess.run(model.tensorDict['p2_emb_focus_matrix'])
        p2_emb_context_matrix = sess.run(model.tensorDict['p2_emb_context_matrix'])
        
        if logger is not None:
            logger.info("\n  (concat 'p1_emb_matrix' and 'p2_emb_focus_matrix')")
        else:
            print("\n  (concat 'p1_emb_matrix' and 'p2_emb_focus_matrix')")
        if not np.isnan(np.sum(p1_emb_matrix)):
            if not np.isnan(np.sum(p2_emb_focus_matrix)):
                emb_matrix = np.concatenate([p1_emb_matrix, p2_emb_focus_matrix], axis=1)
            else:
                if logger is not None:
                    logger.info("\n  (p2_emb_focus_matrix -> nan)")
                else:
                    print("\n  (p2_emb_focus_matrix -> nan)")
                emb_matrix = p1_emb_matrix
                
            filename = model.flag['MODEL_ARCH']+'_'+logPath.split('_')[-1]+'_emb_matrix_{}_{}.pkl'.format(emb_matrix.shape[0],
                                                                                                      emb_matrix.shape[1])
            dumpingFiles(logger, logPath, filename, emb_matrix)
            dumpingFiles(logger, dataSets.info['DATA_FOLDER_PATH'], filename, emb_matrix)
        else:
            if logger is not None:
                logger.info("\n  (p1_emb_matrix -> nan)\n  (ABORT!!)")
            else:
                print("\n  (p1_emb_matrix -> nan)\n  (ABORT!!)")
        
    return [loss_p1, loss_p2]
    
    
def save_model(logger, logPath, sess, g_step):
    import os
    if not os.path.exists(logPath): os.makedirs(logPath)
    tf.train.Saver().save(sess, os.path.join(logPath, os.path.basename(logPath)), global_step=g_step)
    if logger is not None:
        logger.info(" [*] Saving checkpoints... {}".format(logPath))
    else:
        print(" [*] Saving checkpoints... {}".format(logPath))
        
def load_model(logger, logPath, sess):
    import os
    ckpt = tf.train.get_checkpoint_state(os.path.abspath(logPath))
    try:
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
        if logger is not None:
            logger.info("Loading SUCCESS.. ") 
        else:
            print("Loading SUCCESS.. ") 
        return True
    except: 
        if logger is not None:
            logger.info("Loading FAILED.. ")
        else:
            print("Loading FAILED.. ")
        return False

def Train_model_list(MODEL_LIST, DATASETS, DUMPING_PATH, new_game=False):
    """
    Train and Test models
    new_game: if new_game, /RESULT_BASE_PATH/PROJECT_NAME/DB_NAME will be FORMATTED. Initialize saved models and figures.
    """
    import os, datetime
    from .utils import get_logger_instance
    from .report import Run_tensorboard
    
    if not os.path.exists(DUMPING_PATH): 
        os.makedirs(DUMPING_PATH)
        
    #tensorboard
    Run_tensorboard(DUMPING_PATH)
        
    if new_game:
        import shutil, glob, os
        _ = [shutil.rmtree(p) for p in glob.glob(os.path.join(DUMPING_PATH, '**/'))] #remove EMB_MODEL_* folders
        _ = [os.remove(p) for p in glob.glob(os.path.join(DUMPING_PATH, '*.pkl'))]
        _ = [os.remove(p) for p in glob.glob(os.path.join(DUMPING_PATH, '*.html'))]
        #_ = [os.remove(p) for p in glob.glob(os.path.join(DUMPING_PATH, 'MODEL_*.log'))]
        _ = [os.remove(p) for p in glob.glob(os.path.join(DUMPING_PATH, '*_model_list.log'))]
    
    logger = get_logger_instance(logger_name='train_model_list', 
                                 DUMPING_PATH=DUMPING_PATH, 
                                 parent_name='emb_pipeline', 
                                 stream=False)
    logger.info("\n{}".format(datetime.datetime.now()))
    if new_game:
        logger.info("\n(Previous Logs removed)\n")
    logger.info("[Train_model_list]")
    
    RESULTS = []
    for model_idx, model in enumerate(MODEL_LIST):
        logger.info("\n\t[@] MODEL-({}/{}) Training.. \n".format(model_idx+1, len(MODEL_LIST)))
        logger.info("  (model_params)")
        option_printer(logger, **model.flag)
        logPath = os.path.join(DUMPING_PATH, 'EMB_MODEL_'+str(model_idx+1))

        # training
        results = Train_model(logger, DATASETS, model, logPath, new_game)
        
        # collect model_loss
        RESULTS.append([model_idx]+results+[model.flag])
        
    import pandas as pd
    df = pd.DataFrame(RESULTS, columns=['Model_Index', 'Loss_p1', 'Loss_p2', 'Flag'])
    dumpingFiles(logger, DUMPING_PATH, 'df_emb_RESULTS.pkl', df)
    df.to_html(os.path.join(DUMPING_PATH, 'df_emb_RESULTS.html'))
    logger.info("df_emb_RESULTS.html dumped.. {}".format(DUMPING_PATH))
    logger.info("\n[ALL DONE]")
    



