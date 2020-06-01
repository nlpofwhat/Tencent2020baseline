import numpy as np

"""
def make_train_feed_dict(model, batch):
    #make train feed dict for training
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: .5}
    return feed_dict
"""

def make_train_feed_dict(model, batch):
    """make train feed dict for training"""
    """
    print(batch[0][0].shape)
    print(batch[0][1].shape)
    print(batch[0][2].shape)
    print(batch[0][3].shape)
    print(batch[0][4].shape)
    print(batch[0][5].shape)
    print(batch[0][6].shape)
    print(batch[0][7].shape)
    print(batch[1].shape)
    print("\n")
    """
    feed_dict = {model.ad_id: np.array(batch[0][0]),
                 model.advertiser_id: np.array(batch[0][1]),
                 model.click_times: np.array(batch[0][2]),
                 model.creative_id: np.array(batch[0][3]),
                 model.industry: np.array(batch[0][4]),
                 model.product_category: np.array(batch[0][5]),
                 model.product_id: np.array(batch[0][6]),
                 model.time: np.array(batch[0][7]),
                 model.label: batch[1],
                 model.is_training: True}
    """
    feed_dict = {model.ad_id : np.array([x[0] for x in batch[0]]),
                model.advertiser_id : np.array([x[1] for x in batch[0]]),
                model.click_times : np.array([x[2] for x in batch[0]]),
                model.creative_id  : np.array([x[3] for x in batch[0]]),
                model.industry : np.array([x[4] for x in batch[0]]),
                model.product_category : np.array([x[5] for x in batch[0]]),
                model.product_id : np.array([x[6] for x in batch[0]]),
                model.time : np.array([x[7] for x in batch[0]]),
                 model.label: batch[1],
                 model.keep_prob: .5}
    """
    return feed_dict
    
"""
def make_eval_feed_dict(model, batch):
    feed_dict = {model.x: batch[0],
                 model.label: batch[1],
                 model.keep_prob: 1.0}
    return feed_dict
"""
# batch (x_batch, y_batch)
def run_train_step(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'train_op': model.train_op,
        'loss': model.loss,
        'global_step': model.global_step,
        'prediction' : model.prediction
    }
    return sess.run(to_return, feed_dict)

def make_eval_feed_dict(model, batch):
    """make train feed dict for training"""
    """
    print(batch[0][0].shape)
    print(batch[0][1].shape)
    print(batch[0][2].shape)
    print(batch[0][3].shape)
    print(batch[0][4].shape)
    print(batch[0][5].shape)
    print(batch[0][6].shape)
    print(batch[0][7].shape)
    print(batch[1].shape)
    print("\n")
    """
    feed_dict = {model.ad_id: np.array(batch[0][0]),
                 model.advertiser_id: np.array(batch[0][1]),
                 model.click_times: np.array(batch[0][2]),
                 model.creative_id: np.array(batch[0][3]),
                 model.industry: np.array(batch[0][4]),
                 model.product_category: np.array(batch[0][5]),
                 model.product_id: np.array(batch[0][6]),
                 model.time: np.array(batch[0][7]),
                 model.label: batch[1],
                 model.is_training: False}
    """
    feed_dict = {model.ad_id : np.array([x[0] for x in batch[0]]),
                model.advertiser_id : np.array([x[1] for x in batch[0]]),
                model.click_times : np.array([x[2] for x in batch[0]]),
                model.creative_id  : np.array([x[3] for x in batch[0]]),
                model.industry : np.array([x[4] for x in batch[0]]),
                model.product_category : np.array([x[5] for x in batch[0]]),
                model.product_id : np.array([x[6] for x in batch[0]]),
                model.time : np.array([x[7] for x in batch[0]]),
                 model.label: batch[1],
                 model.keep_prob: .5}
    """
    return feed_dict
# 训练集和验证集都需要标签
def run_eval_step(model, sess, batch):
    feed_dict = make_eval_feed_dict(model, batch)
    prediction,label = sess.run([model.prediction,model.label], feed_dict)
    #acc = np.sum(np.equal(prediction, batch[1])) / len(prediction)
    return prediction,label

def make_test_feed_dict(model, batch):
    """make train feed dict for training"""

    feed_dict = {model.ad_id: np.array(batch[0]),
                 model.advertiser_id: np.array(batch[1]),
                 model.click_times: np.array(batch[2]),
                 model.creative_id: np.array(batch[3]),
                 model.industry: np.array(batch[4]),
                 model.product_category: np.array(batch[5]),
                 model.product_id: np.array(batch[6]),
                 model.time: np.array(batch[7]),
                 model.is_training: False}
    return feed_dict
def run_test_step(model, sess, batch):
    feed_dict = make_test_feed_dict(model, batch)
    prediction = sess.run(model.prediction, feed_dict)#prediction为预测的类别
    prediction = prediction + 1
    #acc = np.sum(np.equal(prediction, batch[1])) / len(prediction)
    return prediction



def get_attn_weight(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    return sess.run(model.alpha, feed_dict)
