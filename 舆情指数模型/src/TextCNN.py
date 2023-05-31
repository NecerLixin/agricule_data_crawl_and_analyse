import numpy as np
import torch
import pandas as pd
import json
import torch
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
import gensim

def get_data(file_path):
    with open(file_path,'r') as f:
        data = json.load(f)
    return data
def get_words(data):
    return [i['words'] for i in data]

def get_labels(data):
    return np.array([i['label'] for i in data])

def word_to_num(words_list,word_to_ix:dict):
    res_seq = []
    for words in words_list:
        temp_list = np.array([word_to_ix[i] for i in words])
        res_seq.append(temp_list)
    return res_seq

def seq_pad(sent_list,seq_size=150):
    # 扩展序列
    res_list = []
    for sent in sent_list:
        d = len(sent)
        a = seq_size // len(sent)
        b = seq_size % len(sent)
        # if a == 0:
        #     # b肯定不为0
        #     # seq里面随机取
        #     temp_seq = np.random.choice(sent,seq_size-b)
        #     sent = np.concatenate([sent,temp_seq],axis=0)
        #     print(sent.shape)
        # else:
        temp_seq1 = np.tile(sent,a)
        x1 = len(temp_seq1)
        # print(len(temp_seq1))
        temp_seq2 = np.random.choice(sent,b)
        x2 = len(temp_seq2)

        sent = np.concatenate([temp_seq1,temp_seq2],axis=0)
        z = len(sent)
        res_list.append(sent)
        # continue
    return np.stack(res_list)

def load_pretrained_embedding(word_dict,w2v_model,embedding_dim=100):
    word_list = list(word_dict.keys())
    size = len(word_list)
    emb = torch.zeros([size,embedding_dim])
    for i,word in enumerate(word_list):
        vec = w2v_model.wv[word]
        emb[i,:] = torch.from_numpy(vec)
    return emb

def test_val(model,X_test,y_test):
    model.eval()
    avg_acc = []
    with torch.no_grad():
        for batch,label in zip(X_test,y_test):
            pred = model(batch)
            # loss = criteon(pred,label)
            acc = binary_acc(pred,label).item()
            avg_acc.append(acc)
    res_acc  = np.array(avg_acc).mean()
    print("test_score:",res_acc)
    return res_acc

def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def binary_acc(preds, y):
    """
    get accuracy
    """
    preds = torch.round(preds)
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc

class TextCNN(nn.Module):
    def __init__(self,seq,vocab_size,word_dim) -> None:
        super(TextCNN,self).__init__()
        self.embedding = nn.Embedding(vocab_size,word_dim)
        self.net = nn.Sequential(
            # [batch,1,150,100] -> [batch,3,130,80]
            nn.Conv2d(in_channels=1,out_channels=3,kernel_size=21,stride=1),nn.ReLU(),
            # [batch,3,130,80] -> [batch,3,26,16]
            nn.MaxPool2d(kernel_size=5),
            # [batch,3,26,20] -> [batch,1560]
            nn.Flatten(),
            nn.Linear(3*26*16,200),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(200,20),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(20,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        # [seq,batch,1] -> [seq,batch,1,dim]->[batch,1,seq,dim]
        embedding = self.embedding(x).permute([1,2,0,3])
        out =  self.net(embedding)
        return out

def get_w2v_model(words_list):
    # [[1,1,1,],[1,1]]
    # model.wv['word'] 
    w2v_model = gensim.models.Word2Vec(words_list,vector_size=100,min_count=1,sg=1)
    return w2v_model

def get_word_dict(file_path):
    word_to_ix = get_data(file_path)
    word_dict = dict()
    for key,val in word_to_ix.items():
        word_dict[val] = key
    return word_dict

def binary_acc(preds, y):
    """
    get accuracy
    """
    preds = torch.round(preds)
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc




def train(model,train_set,criteon,optimizer):
    aver_acc = []
    data = train_set[0]
    labels = train_set[1]
    i = 0
    # for batch,label in zip(data,labels):
    for i in range(len(train_set[0])):
        batch = train_set[0][i]
        label = train_set[1][i]
        # break
        i+=1
        pred = model(batch)
        loss = criteon(pred,label)
        acc = binary_acc(pred,label).item()
        aver_acc.append(acc)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%10 == 0:
            print(i, acc)
            # print("loss:",loss.item())
        
    avg_acc = np.array(aver_acc).mean()
    print('avg acc:', avg_acc)

def get_train_eval_set(file_path,word_ix_path,batch_size=20):
    data_all = get_data(file_path)
    word_to_ix = get_data(word_ix_path)
    words_list = get_words(data_all)
    labels = get_labels(data_all)
    sent_list = word_to_num(words_list,word_to_ix)
    seq_list = seq_pad(sent_list,150)
    X = torch.from_numpy(np.stack(seq_list,axis=0))
    y = torch.from_numpy(labels)
    # X,y = get_batch(seq_list,labels,batch_size)
    return train_test_split(X,y,random_state=30)

# X,y = get_train_eval_set('data/data.json','data/word_to_ix.json')
# print(y)
def get_batch1(_data,_label,batch_size):
    # data (n*seq)
    data = torch.from_numpy(_data)
    label = torch.from_numpy(_label)
    a = len(data) // batch_size
    res_data = []
    res_label = []
    for i in range(a-1):
        temp = data[i*batch_size:(i+1)*batch_size].unsqueeze(-1).permute(1,0,2)
        res_data.append(temp)
    res_data.append(data[-batch_size:].unsqueeze(-1).permute(1,0,2))
    for i in range(a-1):
        res_label.append(label[i*batch_size:(i+1)*batch_size].unsqueeze(-1).float())
    res_label.append(label[-batch_size:].unsqueeze(-1).float())
    return res_data,res_label
def get_batch(_X,_y,batch_size):
    random_seq = torch.randperm(_X.shape[0])
    X = _X[random_seq,:]
    y = _y[random_seq]
    res_data = []
    res_label = []
    a = _X.shape[0]//batch_size
    for i in range(a-1):
        # [batch,dim]
        temp = X[i*batch_size:(i+1)*batch_size,:].unsqueeze(-1).permute(1,0,2)
        res_data.append(temp)
    res_data.append(X[-batch_size:,:].unsqueeze(-1).permute(1,0,2))
    for i in range(a-1):
        temp = y[i*batch_size:(i+1)*batch_size].unsqueeze(-1).float()
        res_label.append(temp)
    res_label.append(y[-batch_size:].unsqueeze(-1).float())
    return res_data,res_label
def load_pretrained_embedding(word_dict,w2v_model,embedding_dim=100):
    word_list = list(word_dict.keys())
    size = len(word_list)
    emb = torch.zeros([size,embedding_dim])
    for i,word in enumerate(word_list):
        vec = w2v_model.wv[word]
        emb[i,:] = torch.from_numpy(vec)
        #[word_num,dim]
    return emb


def eval2(rnn,train_set,criteon):
    data = train_set[0]
    labels = train_set[1]
    avg_acc = []
    
    rnn.eval()
    
    with torch.no_grad():
        for batch,label in zip(data,labels):

            # [b, 1] => [b]
            pred = rnn(batch).squeeze(1)

            #
            loss = criteon(pred, label)

            acc = binary_acc(pred, label).item()
            avg_acc.append(acc)
        
    avg_acc = np.array(avg_acc).mean()
    
    print('>>test:', avg_acc)
def test_val(model,X_test,y_test):
    model.eval()
    avg_acc = []
    with torch.no_grad():
        for batch,label in zip(X_test,y_test):
            pred = model(batch)
            # loss = criteon(pred,label)
            acc = binary_acc(pred,label).item()
            avg_acc.append(acc)
    res_acc  = np.array(avg_acc).mean()
    print("test_score:",res_acc)
    return res_acc

def adjust_learning_rate(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def shuffle(_X,_y):
    
    random_seq = torch.randperm(X.shape[0])
    X = X[random_seq,:]
    y = y[random_seq,:]
    return X,y
if __name__ == '__main__':
    data_path = 'data/data.json'
    ix_path = 'data/word_to_ix.json'
    data = get_data(data_path)
    words_list = get_words(data)
    X_train,X_test,y_train,y_test = get_train_eval_set(data_path,ix_path,80)
    w2v_model = get_w2v_model(words_list)
    words_dict = get_word_dict(ix_path)
    pretrained_embedding = load_pretrained_embedding(words_dict,w2v_model)
    # model = lstm(len(words_dict),100,1)
    model = TextCNN(150,len(words_dict),100)
    model.embedding.weight.requires_grad = False
    model.embedding.weight.copy_(pretrained_embedding)
    model.embedding.weight.requires_grad = True
    # model = lstm2(200,100,1,w2v_model,words_dict)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # criteon = nn.BCEWithLogitsLoss()
    criteon = nn.BCELoss()
    # torch.nn.init.kaiming_normal(model.parameters())
    # scheduler = optim.lr_scheduler.LinearLR(optimizer)
    # for i in range(100):
    #     train(model,(X_train,y_train),criteon,optimizer)
    for epoch in range(200):
        # eval2(model, (X_train,y_train), criteon)
        # X_train,y_train = shuffle(X_train,y_train)
        X,y = get_batch(X_train,y_train,80)
        X_t,y_t = get_batch(X_test,y_test,20)
        print(f'epoch:{epoch}---------------------')
        train(model, (X,y), criteon,optimizer)
        test_val(model,X_t,y_t)
        adjust_learning_rate(optimizer,epoch,1e-3)
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        
        
        # for i in y_train:
        #     print(i.shape)
        # break
    torch.save(model,'data/model.pth')
    X_t,y_t = get_batch(X_test,y_test,20)
    test_val(model,X_t,y_t)
    

        