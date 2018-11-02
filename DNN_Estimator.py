# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 23:33:01 2018

@author: hecongcong
"""

'''
用TF.learn构建深度神经网络Estimator.
'''

import tensorflow as tf

####用layers模块建立两个特征列
def _input_fn(num_epochs=None):
    features={'age':tf.train.limit_epochs(tf.constant([[8],[2],[1]]),\
                                        num_epochs=num_epochs),\
    ##产生指定索引指定值的稀疏张量
    '''
    Output:
    [['en','fr'],[0,0],['zh',0]]
    '''
    'language':tf.SparseTensor(values=['en','fr','zh'],\
                               indices=[[0,0],[0,1],[2,0]],\
                               dense_shape=[3,2])}
    return(features,tf.constant([[1],[0],[0]],dtype=tf.int32))
    '''
    Return:x,y.where y represents label's class index.
    '''


language_column=tf.contrib.layers.sparse_column_with_hash_bucket(\
                                            'language',\
                                            hash_bucket_size=20)

feature_columns=[\
    tf.contrib.layers.embedding_column(language_column,dimension=1),\
    tf.contrib.layers.real_valued_column('age')]
'''
real value column:实值列
'''


####把特征列、每层隐藏神经单元数、标识类别等传入DNNClassifier中

'''
classifier:A 'DNNClassifier' estimator.
'''
classifier=tf.contrib.learn.DNNClassifier(\
            '''
            n_classes:
            number of label classes.
            For arbitrary label values(e.g. string labels),
            convert to calss indices first.
            '''
            n_classes=2,\
            feature_columns=feature_columns,\
            hidden_units=[3,3],\ #hidden units per layer
            '''
            config:
            'RunConfig' object to configure the runtime settings.
            '''
            config=tf.contrib.learn.RunConfig(tf_random_seed=1))


####进行模型的训练和评估
classifier.fit(input_fn=_input_fn,steps=100)
scores=classifier.evaluate(input_fn=_input_fn,steps=1)





'''
1.带权重的分类器
'''
####构建权重列和特征列features
def _input_fn_train():
    target=tf.constant([[1],[0],[0],[0]]) #4个sample的labels
    features={\
        'x':tf.ones(shape=[4,1],dtpye=tf.float32),\
        'w':tf.constant([[100.],[3.],[2.],[2.]])}
    return(features,target)


####构建DNNClassifier Estimator
classifier=tf.contrib.learn.DNNClassifier(\
        weight_column_name='w',\
        feature_columns=[tf.contrib.layers.real_valued_column('x')],\
        hidden_units=[3,3],\
        config=tf.contrib.learn.RunConfig(tf_random_seed=3))

classifier.fit(input_fn=_input_fn_train,steps=100)




###############
'''
2:用自定义的metrics方程进行操作
'''
def _input_fn_train():
    target=tf.constant([[1],[0],[0],[0]])
    features={'x':tf.ones(shape=[4,1],dtpye=tf.float32)}
    return(features,target)
    

def _my_metric_op(predictions,targets):
    '''
    predictions:input tensor
    [0,1]:begin location
    [-1,1]:the size of slice tensor
    size[i]=-1:all remaining element in dimension i are 
               includeed in the slice.
    For example:
    input:[[1,2],[3,4],[5,6]]
    output:[[2],[4],[6]]
    '''
    predictions=tf.slice(predictions,[0,1],[-1,1])
    return(tf.reduce_sum(tf.multiply(predictions,targets)))


####构建DNNClassifier Estimator、训练及评估
classifier=tf.contrib.learn.DNNClassifier(\
    feature_columns=[tf.contrib.layers.real_valued_column('x')],\
    hidden_units=[3,3],\
    config=tf.contrib.learn.RunConfig(tf_random_seed=1))


classifier.fit(input_fn=_input_fn_train,steps=100)


scores=classifier.evaluate(\
   input_fn=_input_fn_train,\
   steps=100,\
   '''
   my_accuracy:A tensor representing the accuracy,the value
   of 'total' divided by 'count'.
   
   my_precision:Scalar float 'tensor' with the value of
   'true_positives' divided by the sum of 'true_positives'
   and 'false_positives'.
   '''
   metrics={\
        'my_accuracy':tf.contrib.metrics.streaming_accuracy,\
        ('my_precision','class'):tf.contrib.metrics.streaming_precision,\
        ('my_metric','probabilities'):_my_metric_op})
        


####定义指数递减学习率函数
def optimizer_exp_decay():
    '''
    global_step:全局训练步数
    '''
    global_step=tf.contrib.framework.get_or_create_global_step()
    
    '''
    return:The decayed learning rate.
    
    It is computed as:
    decayed_learning_rate=learning_rate*decay_rate^(global_step/decay_steps)
    '''
    learning_rate=tf.train.exponential_decay(\
            learning_rate=0.1,global_step=global_step,\
            decay_steps=100,decay_rate=0.001)
    
    return(tf.train.AdagradOptimizer(learning_rate=learning_rate))
    

####运用自定义的优化器，构建DNNClassifier及其训练
from sklearn import datasets,cross_validation

iris=datasets.load_iris()

x_train,x_test,y_train,y_test=cross_validation.train_test_split(\
                        iris.data,iris.target,\
                        test_size=0.2,random_state=42)

'''
feature_columns:List of 'FeatureColumn' object.
x_train:real_valued matrix of shape [n_samples,n_features]
'''
feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(\
                                            x_train)

classifier=tf.contrib.learn.DNNClassifer(\
                    feature_columns=feature_columns,\
                    hidden_units=[10,20,10],\
                    n_class=3,\
                    optimizer=optimizer_exp_decay)

classifier.fit(x_train,y_train,steps=800)







   
                