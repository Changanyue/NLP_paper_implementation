# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
#from pytorch_pretrained import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.datasetpkl = dataset + '/data/dataset.pkl'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 64                                         # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768




class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # label_encoder:
        wvdim = 768
        self.label_emb = nn.Embedding(config.num_classes,wvdim) # (n,wvdim)
        self.label_fc = nn.Linear(wvdim, config.hidden_size)
        self.sim_fc = nn.Linear(config.num_classes, config.num_classes)
        self.bert_fc1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.bert_fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids=None, mask=None,l_train=None):

        # seq_output, pooled = self.bert(input_ids, attention_mask=mask, output_all_encoded_layers=False)
        # text_emb = seq_output[:,0,:]
        bert_output = self.bert(input_ids=input_ids, token_type_ids=mask)
        text_emb = bert_output['last_hidden_state'][:,0,:]

        text_emb = torch.tanh(self.bert_fc1(text_emb))
        y_pred = self.bert_fc2(text_emb)

        label_emb = self.label_emb(l_train)
        label_emb = F.tanh(self.label_fc(label_emb))  # [64,768]

        doc_product = torch.bmm(label_emb.squeeze(1), text_emb.unsqueeze(-1))  # (n,d) dot (d,1) --> (n,1)
        #print(doc_product.shape)
        label_sim_dict = self.sim_fc(doc_product.squeeze(-1))
        #print(label_sim_dict.shape)

        return y_pred, label_sim_dict
