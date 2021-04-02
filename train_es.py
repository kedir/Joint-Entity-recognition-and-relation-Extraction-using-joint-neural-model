import utils
import tf_utils
from build_data import build_data
import numpy as np
import tensorflow as tf
import sys
import os.path

'Train the model on the train set and evaluate on the evaluation and test sets until ' \
'(1) maximum epochs limit or (2) early stopping break'
def checkInputs():
    if (len(sys.argv) <= 3) or os.path.isfile(sys.argv[0])==False :
        raise ValueError(
            'The configuration file and the timestamp should be specified.')
def saved_model(sess,input_tensor_dic,output_tensor_dic,output_dir=sys.argv[3]):
    """Step 1. Defines the predict signatures that uses the TF Predict API"""
    ##Method 1
    predict_signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs=input_tensor_dic, outputs=output_tensor_dic)
    ###Method 2
    ##Creates the TensorInfo protobuf objects that encapsulates the input/output tensors
    # input_tensor_infos ={}
    # for key, value in input_tensor_dic.items():
    # input_tensor_infos[key] = tf.saved_model.utils.build_tensor_info(value)
    # output_tensor_infos = {}
    # for key, value in output_tensor_dic.items():
    # output_tensor_infos[key] = tf.saved_model.utils.build_tensor_info(value)
    # #Defines the predict signatures, uses the TF Predict API
    # predict_signature = (
    # tf.saved_model.signature_def_utils.build_signature_def(
    # inputs = input_tensor_infos,
    # outputs = output_tensor_infos,
    # method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    """Step 2. Defines where the model will be exported"""
    # export_path_base = FLAGS.export_model_dir
    export_path_base = '%s/saved_model' % output_dir
    model_version='v1'
    export_path = os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(str(model_version)))
    print('Exporting saved model to', export_path)
    ## Create SavedModelBuilder class
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess=sess, 
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_signature': predict_signature,
        })
    """Step 3. export the model"""
    # builder.save(as_text=True)
    builder.save()
    print('Done exporting!')
if __name__ == "__main__":
    checkInputs()
    config=build_data(sys.argv[1])
    train_data = utils.HeadData(config.train_id_docs, np.arange(len(config.train_id_docs)))
    dev_data = utils.HeadData(config.dev_id_docs, np.arange(len(config.dev_id_docs)))
    test_data = utils.HeadData(config.test_id_docs, np.arange(len(config.test_id_docs)))
    tf.reset_default_graph()
    tf.set_random_seed(1)
    utils.printParameters(config)
    with tf.Session() as sess:
        embedding_matrix = tf.get_variable('embedding_matrix', shape=config.wordvectors.shape, dtype=tf.float32,trainable=False).assign(config.wordvectors)
        emb_mtx = sess.run(embedding_matrix)
        model = tf_utils.model(config,emb_mtx,sess)
        obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel = model.run()
        train_step = model.get_train_op(obj)
        operations=tf_utils.operations(train_step,obj, m_op, predicted_op_ner, actual_op_ner, predicted_op_rel, actual_op_rel, score_op_rel)
        sess.run(tf.global_variables_initializer())
        best_score=0
        nepoch_no_imprv = 0  # for early stopping
        for iter in range(config.nepochs+1):
            model.train(train_data,operations,iter)
            dev_score=model.evaluate(dev_data,operations,'dev')
            model.evaluate(test_data, operations,'test')
            if dev_score>=best_score:
                nepoch_no_imprv = 0
                best_score = dev_score
                print ("- Best dev score {} so far in {} epoch".format(dev_score,iter))
            else:
                nepoch_no_imprv += 1
                if nepoch_no_imprv >= config.nepoch_no_imprv:

                    print ("- early stopping {} epochs without " \
                                     "improvement".format(nepoch_no_imprv))
                    with open(sys.argv[3]+"/es_"+sys.argv[2]+".txt", "w+") as myfile:
                        myfile.write(str(iter))
                        myfile.close()
                    break
        saved_model(sess=sess, 
            input_tensor_dic={
                'charIds': operations.m_op['charIds'],
                'tokensLens':operations.m_op['tokensLens'],
                'embeddingIds': operations.m_op['embeddingIds'],
                'tokenIds': operations.m_op['tokenIds'],
                'tokens': operations.m_op['tokens'],
                'seqlen': operations.m_op['seqlen'],
                'doc_ids': operations.m_op['doc_ids'],
                'isTrain': operations.m_op['isTrain'],
                'entity_tags_ids': operations.m_op['entity_tags_ids'],
                'scoringMatrixGold': operations.m_op['scoringMatrixGold'],
                'dropout_embedding': operations.m_op['dropout_embedding'],
                'dropout_lstm': operations.m_op['dropout_lstm'],
                'dropout_lstm_output': operations.m_op['dropout_lstm_output'],
                'dropout_fcl_ner': operations.m_op['dropout_fcl_ner'],
                'dropout_fcl_rel': operations.m_op['dropout_fcl_rel'],
                'BIO': operations.m_op['doc_ids'],
                'entity_tags': operations.m_op['doc_ids']
            },
            output_tensor_dic={
                'pred_ner_ids': operations.predicted_op_ner,
                'pred_rel_ids': operations.predicted_op_rel,
            },
            output_dir=sys.argv[3]
            )
        model.predict(test_data, operations,'test')