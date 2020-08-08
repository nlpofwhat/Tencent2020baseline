# Tencent2020baseline
 
基于transformer模型,后面可以自己加入pool层

#+++++++++++特征
可以加入以下特征：

ad_id
advertiser_id 
click_times
creative_id 
industry
product_category 
product_id
time

time可以用作position embedding,也可以用delta time作为position embedding，
多篇论文已经证明position embedding对于transformer是很关键的，单纯的统计特征很难对时间序列进行建模

# 主要环境
python 3.5
tensorflow 1.12.0
cuda 9.0
cudnn 7.4

#++++++++++++模型结构从左至右大致如下

embedding  ->  transformer -> concat(max pool ,min pool) ->MLP
总的embedding_size必须能被num_heads=8整除

#++++++++++++samll数据作为调试用

#++++++++++++运行

1. 新建train文件夹把训练集放到里面，新建test文件夹把测试集放到里面
2. python multi_head.py
3. 参数需要自己调整
4. 主程序 python multi_head.py


注意：没有加上k折交叉，单词数据集划分可能导致验证集上的指标不稳定，有待解决
