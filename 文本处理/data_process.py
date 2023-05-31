import pyltp
from pyltp import SentenceSplitter
import jieba
import pandas as pd

data=pd.read_excel("data/data2.xlsx")
# vege_price_dict=pd.read_csv("vege_price.txt")
# vege_price_dict=list(vege_price_dict["word"])

# 分句子
data_size = len(data)
all_data = []
for i in range(data_size):
    title = data.Title[i]
    date = data.Date[i]
    source = data.Source[i]
    sentences = list(SentenceSplitter.split(data.Content[i]))
    for sent in sentences:
        temp_dict = dict()
        temp_dict['title'] = title
        temp_dict['date'] = date
        temp_dict['source'] = source
        temp_dict['sent'] = sent
        all_data.append(temp_dict)
all_data=pd.DataFrame(data=all_data)
all_data.to_excel('data/data_processed2.xlsx',index=False)