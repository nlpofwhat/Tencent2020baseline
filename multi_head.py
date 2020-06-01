from modules.multihead import *
#import matplotlib.pyplot as plt
#from models import *
import numpy as np
from utils.model_helper import *
import time
from utils.prepare_data import *
from sklearn.metrics import confusion_matrix,recall_score
#dropout确实能够提升效果
#函数的原型：numpy.eye(N,M=None,k=0,dtype=<class 'float'>,order='C)
#把标签转化为onehot编码
def to_one_hot(y, n_class):
    return np.eye(n_class)[np.array(y).astype(int)-1]

class AttentionClassifier(object):
    def __init__(self, config):
        self.max_len = config["max_len"]  #300
        self.hidden_size = config["hidden_size"]
        
        # 这里修改7个特制的数目和词向量的维度
        #self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        # 相当于词典大小
        self.ad_id_size = config["ad_id_size"]
        self.advertiser_id_size = config["advertiser_id_size"]
        self.click_times_size = config["click_times_size"]
        self.creative_id_size = config["creative_id_size"]
        self.industry_size = config["industry_size"]
        self.product_category_size = config["product_category_size"]
        self.product_id_size = config["product_id_size"]
        #self.time_size = config["time_size"]
        
        # 相当于词向量大小
        self.ad_embedding_size = config["ad_embedding_size"]
        self.advertiser_embedding_size = config["advertiser_embedding_size"]
        self.click_embedding_size = config["click_embedding_size"]
        self.creative_embedding_size = config["creative_embedding_size"]
        self.industry_embedding_size = config["industry_embedding_size"]
        self.product_category_embedding_size = config["product_category_embedding_size"]
        self.product_id_embedding_size = config["product_id_embedding_size"]
        self.time_embedding_size = config["product_id_size"]
        self.is_training = tf.placeholder(tf.bool)
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

     

        # placeholder
        # 这里需要修改输入特征的占位符
        #self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.ad_id = tf.placeholder(tf.int32, [None, self.max_len], name="input_ad_id")
        self.advertiser_id = tf.placeholder(tf.int32, [None, self.max_len], name="input_advertiser_id")
        self.click_times = tf.placeholder(tf.float32, [None, self.max_len], name="input_click")
        self.creative_id = tf.placeholder(tf.int32, [None, self.max_len], name="input_creative_id")
        self.industry= tf.placeholder(tf.int32, [None, self.max_len], name="input_industry")
        self.product_category = tf.placeholder(tf.int32, [None, self.max_len], name="input_product_category")
        self.product_id = tf.placeholder(tf.int32, [None, self.max_len], name="input_product_id")
        self.time = tf.placeholder(tf.int32, [None, self.max_len], name="input_time")
        
        self.label = tf.placeholder(tf.int32, [None])
        
        self.keep_prob = tf.placeholder(tf.float32)

    def build_graph(self):
        print("building graph...")
        
        # 这里根据嵌入id得到输入的词向量
        self.ad_id_embedding = tf.Variable(
                    tf.random_uniform([self.ad_id_size, self.ad_embedding_size], -1.0, 1.0),
                    name="ad_id")
        self.advertiser_id_embedding = tf.Variable(
            tf.random_uniform([self.advertiser_id_size, self.advertiser_embedding_size], -1.0, 1.0),
            name="advertiser_id")
        
        #self.click_times_embedding = tf.Variable(
        #    tf.random_uniform([self.click_times_size, self.click_embedding_size], -1.0, 1.0),
        #    name="click_times")
        self.creative_id_embedding = tf.Variable(
            tf.random_uniform([self.creative_id_size, self.creative_embedding_size], -1.0, 1.0),
            name="creative_id")
        self.industry_embedding = tf.Variable(
            tf.random_uniform([self.industry_size, self.industry_embedding_size], -1.0, 1.0),
            name="industry")
        self.product_category_embedding = tf.Variable(
            tf.random_uniform([self.product_category_size, self.product_category_embedding_size], -1.0, 1.0),
            name="product_category")
        self.product_id_embedding = tf.Variable(
            tf.random_uniform([self.product_id_size, self.product_id_embedding_size], -1.0, 1.0),
            name="product_id")
        """
        self.time_embedding = tf.Variable(
            tf.random_uniform([self.time_size, self.time_embedding_size], -1.0, 1.0),
            name="time")
        """
        
        self.ad_id_embeddings = tf.nn.embedding_lookup(self.ad_id_embedding, self.ad_id)
        self.advertiser_id_embeddings = tf.nn.embedding_lookup(self.advertiser_id_embedding, self.advertiser_id)
        #self.click_times_embeddings = tf.nn.embedding_lookup(self.click_times_embedding, self.click_times)
        self.click_times_embeddings = tf.expand_dims(self.click_times,axis=-1)
        #batch_size,max_len -> batch_size,max_len,1
        
        self.creative_id_embeddings = tf.nn.embedding_lookup(self.creative_id_embedding, self.creative_id)
        
        self.industry_embeddings = tf.nn.embedding_lookup(self.industry_embedding, self.industry)
        
        self.product_category_embeddings = tf.nn.embedding_lookup(self.product_category_embedding, self.product_category)
        
        self.product_id_embeddings = tf.nn.embedding_lookup(self.product_id_embedding, self.product_id)
        
        #self.time_embeddings = tf.nn.embedding_lookup(self.time_embedding, self.time)
        
        #去掉ad_id self.ad_id_embeddings,
        self.embedded_chars = tf.concat([self.advertiser_id_embeddings ,\
        self.click_times_embeddings,self.creative_id_embeddings,self.industry_embeddings,self.product_category_embeddings,\
        self.product_id_embeddings],axis=-1)
                  
        #embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
        #                             trainable=True)
        #batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)
        

        # multi-head attention
        embedded_chars = positional_encoding(self.embedded_chars,self.time,maxlen=91,masking=True,scope="positional_encoding")
        ma = multihead_attention(queries=embedded_chars, keys=embedded_chars,dropout_rate=0.5，is_training = self.is_training)
        
        # FFN(x) = LN(x + point-wisely NN(x))
        outputs = feedforward(ma, [self.hidden_size, self.embedding_size])
        outputs = tf.reshape(outputs, [-1, self.max_len * self.embedding_size])
        logits = tf.layers.dense(outputs, units=self.n_class)

        # tf.nn.sparse_softmax_cross_entropy_with_logits的label不需要one-hot
        # [0,num_classes)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits+1e-10, labels=self.label-1))
        self.prediction = tf.argmax(tf.nn.softmax(logits), axis=-1)
        #self.y_p = tf.one_hot(self.prediction, 2)
        #self.prediction = to_one_hot(self.prediction,2)
        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")

    # 多分类问题的precision,recall和F1
    def micro_f1(self, pred, label):
        tp = tf.reduce_mean(tf.reduce_sum(pred * label, axis=1))
        fn = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.logical_xor(
            tf.cast(pred, tf.bool), tf.cast(label, tf.bool)
        ), tf.float32) * label, axis=1))
        fp = tf.reduce_mean(tf.reduce_sum(tf.cast(tf.logical_xor(
            tf.cast(pred, tf.bool), tf.cast(label, tf.bool)
        ), tf.float32) * pred, axis=1))
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        return p, r, f1
if __name__ == '__main__':
    # 加入自己的超参数
    config = {
        "max_len": 300,
        "hidden_size": 64,
        #"vocab_size": vocab_size,
        "embedding_size": 80,
        "n_class": 2,  # gender
        #"n_class": 2, #age
        "learning_rate": 0.001,
        "batch_size": 4,#128
        "train_epoch": 20,# 50
        
        #unk可以在读取数据的时候就替换成一个最大的id
        "ad_id_size" : 3812200+1+1, #pad+unk  这个暂时不用
        "advertiser_id_size" : 62965+1+1, #pad+unk
        "click_times_size" : 200+1, #pad
        "creative_id_size" : 4445718+1+1,#pad+unk
        "industry_size" : 335+1+1+1,  # pad+\N+UNK
        "product_category_size" : 18+1+1,#pad+unk
        "product_id_size" : 44313+1+1+1,# pad+\N+UNK
        #self.time_size = config["time_size"]#position embedding
        
        # 相当于词向量大小,这个是超参数具体的维度自己定
        "ad_embedding_size" : 1,
        "advertiser_embedding_size" : 20,
        "click_embedding_size": 1,
        "creative_embedding_size" : 11,
        "industry_embedding_size" : 10,
        "product_category_embedding_size" : 18, #可以使用one-hot
        "product_id_embedding_size" : 20,
        #self.time_embedding_size = 1
    }
    #注意总的embedding_size必须是8的倍数

    #训练集路径
    #ad_file = 'train/ad.csv'
    #click_log_file = 'train/click_log.csv'
    #user_file = 'train/user.csv'
    ad_file = 'small/ad.csv'
    click_log_file = 'small/click_log.csv'
    user_file = 'small/user.csv'
    #n_class=2 gender  age 10
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    x_train, y_train = load_train_data(ad_file,click_log_file,user_file, n_class=2, one_hot=False)
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    
    #测试集路径
    #ad_file = 'test/ad.csv'
    #click_log_file = 'test/click_log.csv'
    #ad_file = 'test/ad.csv'
    #click_log_file = 'test/click_log.csv'
    x_test,user_id = load_test_data(ad_file,click_log_file)
    
    # data preprocessing
    x_train = data_preprocessing_v2(x_train, max_len=config['max_len'])
    x_test = data_preprocessing_v2(x_test, max_len=config['max_len'])
    
    

    # split dataset to test and dev
    # 将训练集划分为8:1的训练集和测试
    print(np.array(x_train).shape)
    print(y_train.shape)

    shuffle_x,shuffle_y = shuffle_me(x_train,y_train)
    x_train, x_dev, y_train, y_dev,dev_size = split_dataset(shuffle_x, shuffle_y, 0.1)
    print("Validation Size: ", dev_size)

    
    classifier = AttentionClassifier(config)
    classifier.build_graph()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    total_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("参数的总数为 :",total_params)
    dev_batch = (x_dev, y_dev)
    start = time.time()
    temp_f1 = 0
    f = open('dev.txt','w')
    for e in range(config["train_epoch"]):

        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        
        y_pred = []
        y_true = []
        i = 0
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, config["batch_size"]):
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))
            prediction = return_dict['prediction']
            y_pred.extend(list(prediction))
            y_true.extend(list(y_batch-1))
            loss = return_dict['loss']
            i= i+1
            if i%4==0:#250
                print("Train batch loss:  %.3f " % loss)
            #print("Train batch acc:  %.3f s" % acc)
        #print(len(y_pred))
        #print(len(y_true))
        
        #print(confusion_matrix(y_true, y_pred).ravel())
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp+fp+1e-9)  # 查准率
        recall = tp / (tp+fn+1e-9)  # 查全率
        f1 = 2*precision*recall/(precision+recall+1e-9)
        print("training precision : %.3f    call : %.3f    f1 : %.3f"%(precision,recall,f1))
        
        #f1 = recall_score(y_true, y_pred, average='micro')
        #print("training precision f1 : %.3f"%(f1))
        
        t1 = time.time()
        y_pred = []
        y_true = []
        
        print("Train Epoch time:  %.3f s" % (t1 - t0))
        print("Start evaluating:  \n")
        t1 = time.time()
        #for x_batch, y_batch in fill_feed_dev_dict(x_dev, y_dev, config["batch_size"]):
        for x_batch, y_batch in fill_feed_dict(x_dev, y_dev, config["batch_size"]):
         
            prediction,y_ture = run_eval_step(classifier, sess, (x_batch, y_batch))
            #prediction = return_dict['prediction']
            y_pred.extend(list(prediction ))
            y_true.extend(list(y_batch-1))
 
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        #print("tn : %.3f  fp : %.3f   fn : %.3f   tp : %.3f"%(tn,fp,fn,tp))
        
        precision = tp / (tp+fp+1e-9)  # 查准率
        recall = tp / (tp+fn+1e-9)  # 查全率
        f2 = 2*precision*recall/(precision+recall+1e-9)
        print("dev precision : %.3f    call : %.3f    f1 : %.3f"%(precision,recall,f2))
        
        #f2 = recall_score(y_true, y_pred, average='micro')
        #print("dev precision f1 : %.3f"%(f2))
        
        t2 = time.time()
        print("dev finished, time consumed : ", t2 - t1, " s")

        """
        #10分类评价指标
        """
        cm = confusion_matrix(y_pred, y_true).ravel()
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        precision = TP / (TP+FP+1e-6)  # 查准率
        recall = TP / (TP+FN+1e-6)  # 查全率
        f2 = 2*precision*recall/(precision+recall+1e-6)
        """      
        if f2=temp_f1 and f1>0.9 :
            temp_f1 = f1
            

            genders = []
            for x_batch in fill_feed_test_dict(x_test, config["batch_size"]):
                prediction = run_test_step(classifier, sess, x_batch)
                genders.extend(prediction)
            
            gender = {'gender':genders}
            df = pd.DataFrame(gender).astype(int) #进行类型转换
            user_id= user_id[['user_id']].drop_duplicates(subset=['user_id'],keep='first',inplace=False)

            user_id = user_id.reset_index(drop=True)  # 重设索引
            #print(user_id)
            #print(df)
            df = pd.concat([user_id,df],axis=1)
            df.to_csv('gender.csv',index=False,sep =',')
        #验证集准确率下降
        #else:
            #if f1_train>0.9:
                #break
    
    
    """
    gender = pd.read_csv('gender.csv',index=Fasle)
    age = pd.read_csv('age.csv',index=Fasle)
    final = pd.merge([age,gender],on='user_id') #合并最后的结果
    final.to_csv('result.csv',index=False)  #生成最后的提交结果
    
    """
    

