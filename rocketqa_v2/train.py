
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from sklearn import metrics
from tqdm import trange, tqdm

from config import set_args
from cross_model import CrossModel
from dual_model import DualModel
from data_utils import SentencePairDataset
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from utils import PGD, FGM
from torch.autograd import Variable


def kl_distance(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), -1)
    return torch.mean(_kl)


def train(dual_model,
          cross_model,
          epochs,
          train_dataloader,
          test_dataloader,
          joint_optimizer,
          joint_scheduler,
          ):
    print("Training epochs: {}".format(epochs))

    # 一轮总共有多少个batch
    batch_num = len(train_dataloader.dataset) / train_dataloader.batch_size
    dual_model.train()
    cross_model.train()


    # if args.use_adv=='fgm':
    #     fgm = FGM(model)
    # if args.use_adv=='pgd':
    #     pgd = PGD(model=model)

    best_dev_loss = 1e+10
    best_dev_f1 = 0.0

    global_step = 0

    train_iterator = trange(int(epochs), desc="Epoch")
    for epoch_idx in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for idx, batch in enumerate(epoch_iterator):

            source_input_ids, target_input_ids, labels, sent_input_ids, att_mask, seg_id = batch

            joint_optimizer.zero_grad()

            # dual model
            dual_features, dual_logits = dual_model(source_input_ids, target_input_ids)
            dual_loss = criterion(dual_logits, labels.squeeze(1))
            label = labels.cpu().numpy().tolist()
            dual_pred = dual_logits.detach().argmax(axis=1).cpu().numpy().tolist()

            # cross-model output
            cross_features, cross_logits = cross_model(sent_input_ids, att_mask, seg_id)
            cross_pred = cross_logits.detach().argmax(axis=1).cpu().numpy().tolist()
            cross_loss = criterion(cross_logits, labels.squeeze(1))

            # KLloss
            KLloss = kl_distance(dual_logits.view(-1), cross_logits.view(-1))
            joint_loss = 10 * KLloss + cross_loss + dual_loss
            # joint_loss = cross_loss

            joint_loss.backward()   # retain_graph=True
            joint_optimizer.step()
            joint_scheduler.step()

            global_step += 1

            if global_step % 50 == 0:
                print('epoch:{}, global_step:{}, loss:{}, KLloss: {}, dual_loss: {}, cross_loss: {}'.format(
                    epoch_idx, global_step, joint_loss.item(), KLloss, dual_loss, cross_loss,
                ))

            if global_step % 300 == 0:
                # 多少步验证依次 一轮进行两次验证，中间一次和最后一次
                dev_loss, dev_acc, dev_f1 = dual_eval(dual_model, test_dataloader)

                if dev_f1 > best_dev_f1:
                    best_dev_loss = dev_loss
                    best_dev_f1 = dev_f1

                    print('epoch:{}, global_step:{}, best_dev_loss:{}, best_dev_f1: {}, '.format(
                        epoch_idx, global_step, best_dev_loss, best_dev_f1,
                    ))

                    torch.save(dual_model.state_dict(),
                               args.save_dir + '_step_{}'.format(global_step) + 'loss')

                dev_loss_1, dev_acc_1, dev_f1_1 = cross_eval(cross_model, test_dataloader)
                print('Cross model: epoch:{}, global_step:{}, dev_loss_1:{}, dev_f1_1: {}, '.format(
                    epoch_idx, global_step, dev_loss_1, dev_f1_1,
                ))


    return best_dev_loss, best_dev_f1


def dual_eval(model, test_dataloader):
    print("Evaluating")
    model.eval()

    total_loss = []
    total_label, total_pred = [], []

    for idx, batch in tqdm(enumerate(test_dataloader)):
        source_input_ids, target_input_ids, labels, sent_input_ids, att_mask, seg_id = batch
        labels = labels.view(-1).cuda()

        with torch.no_grad():
            _, output = model(source_input_ids, target_input_ids)

            loss = criterion(output, labels)

            label = labels.cpu().numpy().tolist()
            pred = output.argmax(axis=1).cpu().numpy().tolist()

            total_pred += pred
            total_label += label

            total_loss.append(loss.item())

    loss = np.array(total_loss).mean()
    acc = metrics.accuracy_score(total_label, total_pred) if len(total_label) != 0 else 0
    f1 = metrics.f1_score(total_label, total_pred, zero_division=0)

    print("Loss on dev set: ", loss)
    print("F1 on dev set: {:.6f}".format(f1))

    return loss, acc, f1


def cross_eval(model, test_dataloader):
    print("Evaluating")
    model.eval()

    total_loss = []
    total_label, total_pred = [], []

    for idx, batch in tqdm(enumerate(test_dataloader)):
        source_input_ids, target_input_ids, labels, sent_input_ids, att_mask, seg_id = batch
        labels = labels.view(-1).cuda()

        with torch.no_grad():
            _, output = model(sent_input_ids, att_mask, seg_id)

            loss = criterion(output, labels)

            label = labels.cpu().numpy().tolist()
            pred = output.argmax(axis=1).cpu().numpy().tolist()

            total_pred += pred
            total_label += label

            total_loss.append(loss.item())

    loss = np.array(total_loss).mean()
    acc = metrics.accuracy_score(total_label, total_pred) if len(total_label) != 0 else 0
    f1 = metrics.f1_score(total_label, total_pred, zero_division=0)

    print("Loss on dev set: ", loss)
    print("F1 on dev set: {:.6f}".format(f1))

    return loss, acc, f1


if __name__ == '__main__':
    args = set_args()

    data_dir = 'ccks'
    train_data_dir, dev_data_dir = [], []

    train_data_dir.append(args.data_dir + data_dir + '/train.txt')
    dev_data_dir.append(args.data_dir + data_dir + '/valid.txt')


    dual_model = DualModel().cuda()
    cross_model = CrossModel().cuda()

    # 加载数据集
    train_dataset = SentencePairDataset(train_data_dir, is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    dev_dataset = SentencePairDataset(dev_data_dir, is_train=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # grouped params
    optimizer_grouped_parameters = []
    no_decay = ["bias", "LayerNorm.weight"]
    dual_model_params = list(dual_model.named_parameters())
    optimizer_grouped_parameters += [
        {'params': [p for n, p in dual_model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in dual_model_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    cross_model_params = list(cross_model.named_parameters())
    optimizer_grouped_parameters += [
        {'params': [p for n, p in cross_model_params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in cross_model_params if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]

    joint_optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    # 总的训练步数
    total_steps = len(train_dataloader) * args.epochs

    # 线性学习率衰减
    joint_scheduler = get_linear_schedule_with_warmup(
        joint_optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=0.05 * total_steps,  # 预热多少步
    )

    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    best_dev_loss = 999
    best_dev_f1 = 0

    train_loss, train_f1 = train(
        dual_model, cross_model, args.epochs,
        train_dataloader, dev_dataloader,
        joint_optimizer, joint_scheduler,
    )
