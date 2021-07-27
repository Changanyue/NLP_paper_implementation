import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BERT_LCM(nn.Module):
    def __init__(self,pretrained_model_name_or_path,hidden_size,num_classes,alpha,wvdim=768,max_len=128,label_embedding_matrix=None):
        super(BERT_LCM, self).__init__()

        self.num_classes = num_classes

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.bert_fc1 = nn.Linear(self.bert.config.hidden_size, hidden_size)
        self.bert_fc2 = nn.Linear(hidden_size, num_classes)

        # label_encoder:
        if label_embedding_matrix is None: # 不使用pretrained embedding
            self.label_emb = nn.Embedding(num_classes,wvdim) # (n,wvdim)
        else:
            self.label_emb = nn.Embedding(num_classes,wvdim,_weight=label_embedding_matrix)
        self.label_fc = nn.Linear(wvdim, hidden_size)

        self.sim_fc = nn.Linear(num_classes, num_classes)

    def forward(self, input_ids=None, token_type_ids=None, labels=None):
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids)
        text_emb = bert_output['last_hidden_state'][:,0,:]
        text_emb = torch.tanh(self.bert_fc1(text_emb))
        # print(text_emb.shape,'text')  # [16,64]
        y_pred = self.bert_fc2(text_emb)

        label_emb = self.label_emb(labels)
        label_emb = F.tanh(self.label_fc(label_emb))
        # print(label_emb.shape,'label')  # [16,20,64]

        doc_product = torch.bmm(label_emb, text_emb.unsqueeze(-1))  # (b,n,d) dot (b,d,1) --> (b,n,1)
        # print(doc_product.shape)   # [16,20,1]
        label_sim_dict = self.sim_fc(doc_product.squeeze(-1))
        #print(label_sim_dict.shape)

        return y_pred, label_sim_dict

