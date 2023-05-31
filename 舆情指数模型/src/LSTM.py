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

def seq_np_to_torch(seq_list):
    # 将np序列转换为tensor
    pass

class lstm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        # vocab_size = 2000
        # embedding_dim = 100
        """
        super(lstm, self).__init__()
        
        # [0-10001] => [100]
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # [100] => [256]
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, 
                           bidirectional=True)
        # [256*2] => [1]
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        """
        x: [seq_len, b,100] vs [b, 3, 28, 28]
        """
        # [seq, b, 1] => [seq, b, 1,100] -> [seq,b,100]
        embedding = self.dropout(self.embedding(x).squeeze(2))
        # embedding = self.embedding(x).squeeze(2)
        
        # output: [seq, b, hid_dim*2]
        # hidden/h: [num_layers*2, b, hid_dim]
        # cell/c: [num_layers*2, b, hid_di]
        output, (hidden, cell) = self.rnn(embedding)
        
        # [num_layers*2, b, hid_dim] => 2 of [b, hid_dim] => [b, hid_dim*2]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # [b, hid_dim*2] => [b, 1]
        hidden = self.dropout(hidden)
        s = self.fc(hidden)
        out = self.sigmoid(s)
        return out

class lstm2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,w2v_model,words_dict):
        """
        需要words_dict,w2v_model
        """
        super(lstm2, self).__init__()
        
        # [0-10001] => [100]
        # 词嵌入
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.words_dict = words_dict
        self.w2v_model = w2v_model
        # [100] => [256]
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, 
                           bidirectional=True, dropout=0.5)
        # [256*2] => [1]
        self.fc = nn.Linear(hidden_dim*2, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
    def embedding2(self,x:torch.Tensor):
        res = []
        # x_ [b,seq_len]
        x_ = x.squeeze(-1).permute([1,0])
        batch = []
        for seq in x_:
            word_list = [self.words_dict[i] for i in seq.tolist()]
            # [seq_len,100]
            seq_vector = np.array([w2v_model.wv[word] for word in word_list])
            batch.append(seq_vector)
        batch = np.stack(batch)
        # [batch,seq_len,100]
        batch = torch.from_numpy(batch).permute([1,0,2])
        return batch
        
    def forward(self,x):
        """
        x: [seq_len, b] vs [b, 3, 28, 28]
        """
        # [seq, b, 1] => [seq, b, 100]
        # embedding = self.dropout(self.embedding(x).squeeze(2))
        embedding = self.embedding2(x)
        a = embedding.shape
        
        # output: [seq, b, hid_dim*2]
        # hidden/h: [num_layers*2, b, hid_dim]
        # cell/c: [num_layers*2, b, hid_di]
        output, (hidden, cell) = self.rnn(embedding)
        
        # [num_layers*2, b, hid_dim] => 2 of [b, hid_dim] => [b, hid_dim*2]
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # [b, hid_dim*2] => [b, 1]
        hidden = self.dropout(hidden)
        s = self.fc(hidden)
        out = self.sigmoid(s)
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
    model = lstm(len(words_dict),100,1)
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


