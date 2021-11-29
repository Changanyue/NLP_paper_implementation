
import torch
import math
from torch import nn
from pdb import set_trace
from NEZHA.model_nezha import NEZHAModel, NezhaConfig
from NEZHA import nezha_utils

class DualModel(nn.Module):
    def __init__(self):
        super(DualModel, self).__init__()
        self.nezha_config = NezhaConfig.from_json_file('./model_weight/NEZHA/config.json')
        self.nezha_model = NEZHAModel(config=self.nezha_config)
        nezha_utils.torch_init_model(self.nezha_model, './model_weight/NEZHA/pytorch_model.bin')

        hidden_size = 768
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_size * 4, 2)


    def forward(self, source_input_ids, target_input_ids):
        source_attention_mask = torch.ne(source_input_ids, 0)  # size: batch_size, max_len
        target_attention_mask = torch.ne(target_input_ids, 0)

        source_embedding = self.nezha_model(source_input_ids, attention_mask=source_attention_mask)
        target_embedding = self.nezha_model(target_input_ids, attention_mask=target_attention_mask)

        source_pooled_embedding = source_embedding[1]
        target_pooled_embedding = target_embedding[1]

        features = torch.cat(
            [
                source_pooled_embedding,
                target_pooled_embedding,
                source_pooled_embedding * target_pooled_embedding,
                torch.abs(source_pooled_embedding - target_pooled_embedding),
            ],
            dim=-1
        )
        logits = self.classifier(features)

        return None, logits

