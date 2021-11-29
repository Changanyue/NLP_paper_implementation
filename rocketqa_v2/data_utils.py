
import torch
import json
import pandas as pd
from tqdm import tqdm
from config import set_args
from torch.utils.data.dataset import Dataset
from transformers.models.bert import BertTokenizer


args = set_args()


class SentencePairDataset(Dataset):
    def __init__(self, file_dir, is_train=True):
        # file_dir 多个数据的路径列表
        self.is_train = is_train
        self.len_limit = args.len_limit
        self.joint_len_limit = args.joint_len_limit
        self.total_source_input_ids = []
        self.total_target_input_ids = []
        self.total_sent_input_ids = []
        self.total_segment_ids = []

        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)

        lines = []
        for file in file_dir:
            with open(file, 'r', encoding='utf8') as f:
                for item in f.readlines():
                    if "{" not in item and "}" not in item:
                        # txt 格式

                        item = item.strip()
                        if not item:
                            continue

                        item = item.split("\t")
                        assert len(item) == 3
                        line = {
                            "source": item[0],
                            "target": item[1],
                            "label": item[2],
                        }

                    else:

                        line = json.loads(item.strip())
                        if 'labelA' in line:
                            line['label'] = line['labelA']
                            del line['labelA']
                        else:
                            line['label'] = line['labelB']
                            del line['labelB']

                    lines.append(line)

        content = pd.DataFrame(lines)
        content.columns = ['source', 'target', 'label']

        sources = content['source'].values.tolist()
        targets = content['target'].values.tolist()

        if self.is_train:
            self.labels = content['label'].values.tolist()
        else:
            self.ids = content['label'].values.tolist()

        for source, target in tqdm(zip(sources, targets), total=len(sources)):
            source = tokenizer.encode(source)[1:-1]   # 去掉[CLS] 和 [SEP]
            target = tokenizer.encode(target)[1:-1]


            if len(source) > (self.joint_len_limit - 3)//2:
                source = source[0:(self.joint_len_limit - 3)//2]
            if len(target) > (self.joint_len_limit - 3)//2:
                target = target[0:(self.joint_len_limit - 3)//2]


            sent = [101] + source + [102] + target + [102]
            seg_id = [0]*(len(source) + 2) + [1]*(len(target) + 1)
            assert len(sent) <= self.joint_len_limit, 'len error1'

            self.total_sent_input_ids.append(sent)
            self.total_segment_ids.append(seg_id)

            if len(source) + 2 > self.len_limit:
                source = source[-self.len_limit + 2:]
            if len(target) + 2 > self.len_limit:
                target = target[-self.len_limit + 2:]

            # 检查序列有没有超过限制
            assert len(source)+2 <= self.len_limit and len(target) + 2 <= self.len_limit

            # [CLS]:101, [SEP]:102
            source_input_ids = [101] + source + [102]
            target_input_ids = [101] + target + [102]

            assert len(source_input_ids) <= self.len_limit and len(target_input_ids) <= self.len_limit

            self.total_source_input_ids.append(source_input_ids)
            self.total_target_input_ids.append(target_input_ids)


    def __len__(self):
        return len(self.total_target_input_ids)

    def __getitem__(self, idx):
        source_input_ids = pad_for_dual_tower(self.total_source_input_ids[idx], self.len_limit+2)
        target_input_ids = pad_for_dual_tower(self.total_target_input_ids[idx], self.len_limit+2)


        sent_input_ids, att_mask, segment_ids = pad_for_cross(self.total_sent_input_ids[idx],self.total_segment_ids[idx], self.joint_len_limit+3)  #self.len_limit2+3

        if self.is_train:
            label = int(self.labels[idx])
            return torch.LongTensor(source_input_ids).cuda(), torch.LongTensor(target_input_ids).cuda(), torch.LongTensor([label]).cuda()\
                , torch.LongTensor(sent_input_ids).cuda(), torch.LongTensor(att_mask).cuda(), torch.LongTensor(segment_ids).cuda()
        else:
            index = self.ids[idx]
            return torch.LongTensor(source_input_ids).cuda(), torch.LongTensor(target_input_ids).cuda(), index\
                , torch.LongTensor(sent_input_ids).cuda(), torch.LongTensor(att_mask).cuda(), torch.LongTensor(segment_ids).cuda()



def pad_for_dual_tower(input_ids, max_len, pad_value=0):
    if len(input_ids) >= max_len:
        input_ids = input_ids[:max_len]
    else:
        input_ids = input_ids + [pad_value] * (max_len - len(input_ids))
    return input_ids

def pad_for_cross(input_ids,segment_ids, max_len):

    if len(input_ids) == max_len:
        att_mask = [1] * len(input_ids)
        seg_ids = segment_ids
    elif len(input_ids) > max_len:
        print('len error2')
        att_mask = []
    elif len(input_ids) < max_len:
        att_mask = [1] * len(input_ids) + [0] * (max_len - len(input_ids))
        seg_ids = segment_ids + [0] * (max_len - len(input_ids))

        input_ids = input_ids + [0] * (max_len-len(input_ids))

    return input_ids, att_mask, seg_ids


