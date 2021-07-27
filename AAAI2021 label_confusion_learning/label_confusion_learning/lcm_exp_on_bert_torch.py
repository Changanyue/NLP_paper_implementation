import os
import time
import datetime
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from utils_torch import load_dataset, create_asy_noise_labels
from models.bert_torch import BERT_LCM


#%%
# ========== parameters: ==========
maxlen = 128
wvdim = 256
hidden_size = 64
batch_size = 16
epochs = 16
lcm_stop = 8

# ========== bert config: ==========
# text_encoder:
tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')

# ========== dataset: ==========
dataset_name = '20NG'
group_noise_rate = 0.3
df,num_classes,label_groups = load_dataset(dataset_name)
# define log file name:
log_txt_name = '%s_BERT_log(group_noise=%s,comp+rec+talk)' % (dataset_name,group_noise_rate)

df = df.dropna(axis=0,how='any')
df = shuffle(df)[:50000]
print('data size:',len(df))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
# ========== data preparation: ==========
labels = sorted(list(set(df.label)))
assert len(labels) == num_classes,'wrong num of classes!'
label2idx = {name:i for name,i in zip(labels,range(num_classes))}

if os.path.exists('data.pkl'):
    (X_token, X_seg, y) = joblib.load('data.pkl')
else:
    #%%
    print('start tokenizing...')
    t = time.time()
    X_token = []
    X_seg = []
    y = []
    i = 0
    for content,label in zip(list(df.content),list(df.label)):
        i += 1
        if i%1000 == 0:
            print(i)
        encoeded = tokenizer.encode_plus(content, max_length=maxlen, truncation=True, padding='max_length', add_special_tokens=True)
        token_ids, seg_ids = encoeded['input_ids'], encoeded["token_type_ids"]

        X_token.append(token_ids)
        X_seg.append(seg_ids)
        y.append(label2idx[label])
    X_token = np.array(X_token)
    X_seg = np.array(X_seg)
    y = np.array(y)

    joblib.dump((X_token, X_seg, y), 'data.pkl')

    print('tokenizing time cost:',time.time()-t,'s.')


#%%
class CustomDataset(Dataset):
    def __init__(self, input_ids, token_type_ids, labels, y_true):
        super(CustomDataset, self).__init__()

        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.labels = labels
        self.y_true = y_true

    def __len__(self):
        return len(self.y_true)

    def __getitem__(self, i):
        return self.input_ids[i], self.token_type_ids[i], self.labels[i], self.y_true[i]

#%%
def lcm_loss(y_true,y_pred,label_sim_dist,alpha):
    label_sim_dist = F.softmax(label_sim_dist)
    pred_log_probs = F.log_softmax(y_pred)
    simulated_y_true = F.softmax(label_sim_dist + alpha * F.one_hot(y_true, num_classes))
    loss = nn.KLDivLoss()(pred_log_probs, simulated_y_true)
    return loss

def lcm_evaluate(model,loader):
    model.eval()
    all_predictions = []
    all_y_true = []
    with torch.no_grad():
        for (X_token_batch,X_seg_batch,L_batch,y_batch) in loader:
            X_token_batch = X_token_batch.to(device)
            X_seg_batch = X_seg_batch.to(device)
            L_batch = L_batch.long().to(device)
            y_batch = y_batch.to(device)

            y_pred, label_sim_dict = model(X_token_batch, X_seg_batch, L_batch)

            y_pred = y_pred.detach().cpu().tolist()
            predictions = np.argmax(y_pred,axis=1)
            all_predictions.extend(predictions)

            y_batch = y_batch.detach().cpu().tolist()
            all_y_true.extend(y_batch)

        acc = round(accuracy_score(all_y_true,all_predictions),5)

        return acc

def train_one_epoch(model, train_loader, optimizer, alpha, loss_type):
    model.train()
    for (X_token_batch,X_seg_batch,L_batch,y_batch) in train_loader:
        X_token_batch = X_token_batch.to(device)
        X_seg_batch = X_seg_batch.to(device)
        L_batch = L_batch.long().to(device)
        print(L_batch.shape)
        y_batch = y_batch.to(device)

        y_pred, label_sim_dict = model(X_token_batch, X_seg_batch, L_batch)

        if loss_type == 'lcm':
            loss = lcm_loss(y_batch, y_pred, label_sim_dict, alpha)
        else:
            loss = nn.CrossEntropyLoss()(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print(loss.item())

def train_val(data_package, batch_size,epochs,alpha,lcm_stop=50,save_best=False):
    model = BERT_LCM('./bert-base-uncased',hidden_size,num_classes,alpha,wvdim,maxlen).to(device)
    optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1.0)

    X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package

    L_train = np.array([np.array(range(num_classes)) for i in range(len(X_token_train))])
    L_val = np.array([np.array(range(num_classes)) for i in range(len(X_token_val))])

    L_test = np.array([np.array(range(num_classes)) for i in range(len(X_token_test))])

    # print(X_token_train.shape, L_train.shape)  (11292, 128) (11292, 20)
    train_dataset = CustomDataset(X_token_train, X_seg_train, L_train, y_train)
    val_dataset = CustomDataset(X_token_val, X_seg_val, L_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_score = 0
    test_score = 0
    train_score_list = []
    val_socre_list = []

    """实验说明：
    每一轮train完，在val上测试，记录其accuracy，
    每当val-acc达到新高，就立马在test上测试，得到test-acc，
    这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
    """

    for i in range(epochs):
        t1 = time.time()
        if i < lcm_stop:
            train_one_epoch(model, train_loader, optimizer, alpha, 'lcm')

            # record train set result:
            train_score = lcm_evaluate(model, train_loader)
            train_score_list.append(train_score)

            # validation:
            val_score = lcm_evaluate(model, val_loader)
            val_socre_list.append(val_score)

            t2 = time.time()
            print('(LCM)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                  val_score)

            # save best model according to validation & test result:
            if val_score > best_val_score:
                best_val_score = val_score
                print('Current Best model!', 'current epoch:', i + 1)

                # test on best model:
                test_score = lcm_evaluate(model, val_loader)
                print('  Current Best model! Test score:', test_score)
                if save_best:
                    torch.save(model.state_dict(), 'best_model_bert_lcm.pt')
                    print('  best model saved!')
        else:
            train_one_epoch(model, train_loader, optimizer, alpha, 'cross_entropy')

            # record train set result:
            train_score = lcm_evaluate(model, train_loader)
            train_score_list.append(train_score)

            # validation:
            val_score = lcm_evaluate(model, val_loader)
            val_socre_list.append(val_score)
            t2 = time.time()
            print('(LCM_stopped)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                  val_score)

            # save best model according to validation & test result:
            if val_score > best_val_score:
                best_val_score = val_score
                print('Current Best model!', 'current epoch:', i + 1)

                # test on best model:
                test_score = lcm_evaluate(model, val_loader)
                print('  Current Best model! Test score:', test_score)

                if save_best:
                    torch.save(model.state_dict(), 'best_model_bert_lcm.pt')
                    print('  best model saved!')
    return train_score_list, val_socre_list, best_val_score, test_score

#%%
# ========== model traing: ==========
old_list = []
ls_list = []
lcm_list = []
N = 5
for n in range(N):
    # randomly split train and test each time:
    np.random.seed(n) # 这样保证了每次试验的seed一致
    random_indexs = np.random.permutation(range(len(X_token)))
    train_size = int(len(X_token)*0.6)
    val_size = int(len(X_token)*0.15)

    X_token_train = X_token[random_indexs][:train_size]
    X_token_val = X_token[random_indexs][train_size:train_size+val_size]
    X_token_test = X_token[random_indexs][train_size+val_size:]
    X_seg_train = X_seg[random_indexs][:train_size]
    X_seg_val = X_seg[random_indexs][train_size:train_size + val_size]
    X_seg_test = X_seg[random_indexs][train_size + val_size:]
    y_train = y[random_indexs][:train_size]
    y_val = y[random_indexs][train_size:train_size+val_size]
    y_test = y[random_indexs][train_size+val_size:]
    data_package = [X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test]

    # apply noise only on train set:
    if group_noise_rate>0:
        _, overall_noise_rate, y_train = create_asy_noise_labels(y_train,label_groups,label2idx,group_noise_rate)
        data_package = [X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test,
                        y_test]
        with open('output/%s.txt' % log_txt_name, 'a') as f:
            print('-'*30,'\nNOITCE: overall_noise_rate=%s'%round(overall_noise_rate,2), file=f)

    with open('output/%s.txt'%log_txt_name,'a') as f:
        print('\n',str(datetime.datetime.now()),file=f)
        print('\n ROUND & SEED = ',n,'-'*20,file=f)

    print('====LCM:============')
    # alpha = 3
    for alpha in [3,4,5]:
        params_str = 'a=%s, wvdim=%s, lcm_stop=%s'%(alpha,wvdim,lcm_stop)

        train_score_list, val_socre_list, best_val_score, test_score = train_val(data_package, batch_size, epochs, alpha, lcm_stop)

        plt.plot(train_score_list, label='train')
        plt.plot(val_socre_list, label='val')
        plt.title('BERT with LCM')
        plt.legend()
        #plt.show()
        plt.savefig(f'alpha={alpha}.png')

        old_list.append(test_score)
        with open('output/%s.txt'%log_txt_name,'a') as f:
            print('\n*** Orig BERT with LCM (%s) ***:'%params_str,file=f)
            print('test acc:', str(test_score), file=f)
            print('best val acc:', str(best_val_score), file=f)
            print('train acc list:\n', str(train_score_list), file=f)
            print('val acc list:\n', str(val_socre_list), '\n', file=f)
