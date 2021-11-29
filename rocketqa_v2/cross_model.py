
import torch
import math
from torch import nn
from pdb import set_trace
from NEZHA.model_nezha import NEZHAModel, NezhaConfig
from NEZHA import nezha_utils


class CrossModel(nn.Module):
    def __init__(self):
        super(CrossModel, self).__init__()
        self.nezha_config = NezhaConfig.from_json_file('./model_weight/NEZHA/config.json')
        self.nezha_model = NEZHAModel(config=self.nezha_config)
        self.classifier = nn.Linear(768, 2)
        nezha_utils.torch_init_model(self.nezha_model, './model_weight/NEZHA/pytorch_model.bin')

    def forward(self, input_ids, att_mask, segment_id):

        # get bert output
        seq_out, pooled_out = self.nezha_model(input_ids, attention_mask=att_mask, token_type_ids=segment_id)
        logits = self.classifier(pooled_out)

        return pooled_out, logits




