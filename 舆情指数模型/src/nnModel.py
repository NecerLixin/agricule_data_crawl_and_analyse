import torch
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(file_path):
    model = torch.load(file_path)
    return model

def get_train_eval_set(file_path,word_ix_path,batch_size=20):
    data_all = get_data(file_path)
    word_to_ix = get_data(word_ix_path)
    words_list = get_words(data_all)
    labels = get_labels(data_all)
    sent_list = word_to_num(words_list,word_to_ix)
    seq_list = seq_pad(sent_list,150)
    X = torch.from_numpy(np.stack(seq_list,axis=0)).to(device)
    y = torch.from_numpy(labels).to(device)
    # X,y = get_batch(seq_list,labels,batch_size)
    return train_test_split(X,y,random_state=30)