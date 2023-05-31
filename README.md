# agricultural_data_crawl_and_analyse
# 文件夹介绍
## 1 爬虫
  主要使用到selenium进行可视化爬虫，使用firefox浏览器，geckodriver，爬取的平台为搜狗微信
  
  验证码识别使用超级鹰
## 2 文本处理
  使用哈工大pyltp工具进行分句，然后使用jieba进行分词，去停词，后续是对分割的句子进行标注，价格上升标注为1，下降标注为-1，不涉及价格变化标注为0（后续去掉），文件里面貌似还有使用机器学习进行分类的代码，使用TFIDF特征。
## 3 舆情指数
  两个模型，一个LSTM，一个TextCNN，第一次写神经网络，加载数据集没有使用Dateset类，写得有点丑。词嵌入Embedding使用的word2vec，但是只用爬取的文本进行了训练，没有引入其他文本进行训练。
