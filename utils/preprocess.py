import numpy as np
import pandas as pd
import os



train_file = 'train_preliminary'
test_file = 'test'

def pre(*,input_path,output_path):
    ad_path = os.join(file_path,'/ad_id.csv')
    click_log_path = os.join(file_path,'/click_log.csv')
    ad = pd.read_csv(ad_path,'r')
    click_log = pd.read_csv(click_log_path,'r')
    ad_click = pd.merge(ad,click_log,on='creative_id')
    ad_click['industry'] = ad_click['industry'].astype(str).replace(r'\N','336').astype(int)
    #test_ad_click['industry'] = test_ad_click['industry'].apply(lambda x: 0 if str(x)==r'\N' else int(str(x)))
    ad_click['product_id'] =ad_click['product_id'].astype(str).replace(r'\N','0').astype(int)
    ad_click = ad_click.sort_values(by=['user_id','time'], ascending=[True,True])
    