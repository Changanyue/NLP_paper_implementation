# -*- coding: utf-8 -*-
import torch
import numpy as np
from network import HGraphConvolutionNet
import torch.nn as nn
import torch.optim as optim
import os
import gc
import matplotlib.pyplot as plt
import time

# 固定随机数种子，方便复现结果
np.random.seed(1)
torch.random.manual_seed(1)
torch.set_default_tensor_type(torch.FloatTensor)

# 是否使用GPU
gpu = True

# 默认数据路径和模型保存路径
fast_snippet_path = 'data/snippets.pt'
snippet_path = 'data/snippets.npz'
best_model_path = 'model/best_model.pt'

# 如果训练时GPU不可用，则使用CPU
train_device = 'cuda' if (torch.cuda.is_available() and gpu) else 'cpu'
test_device = 'cpu' # 强制使用CPU进行测试

# 训练参数
learning_rate = 0.005
weight_decay = 5e-6
max_epochs = 1000
num_per_class = 100 # 训练集和验证集中每一类别的样本总和，训练时训练集和验证集各占一半，如100的话，训练集每一类别有50个，验证集每一类别有50个

# 网络参数
num_hiddens = [512]
num_layer = 2

def block_diag(dd, dt, de, tt, te, ee):
    """
    组合特征矩阵，输出 N*N矩阵

    """
    _num_document = len(dd)
    _num_topic = len(tt)
    _num_entity = len(ee)
    _part_1 = torch.cat([dd, dt, de], dim=1)
    _part_2 = torch.cat([dt.T, tt, te], dim=1)
    _part_3 = torch.cat([de.T, te.T, ee], dim=1)
    return torch.cat([_part_1, _part_2, _part_3])


def normalize_adj(adj, device):
    """
    计算归一化的邻接矩阵，具体内容参考论文中公式1

    """
    n_samples = adj.shape[0]
    adj_norm = adj + torch.eye(n_samples).to(device)
    m = torch.diag(torch.sqrt(torch.tensor(1).to(device) / torch.sum(adj_norm, dim=1)))
    adj_norm = torch.matmul(torch.matmul(m, adj_norm), m)
    return adj_norm


def block_split(num_document, num_topic, num_entity):
    """
    辅助函数，生成各类型节点在邻接矩阵中的索引

    """
    x_1 = num_document
    x_2 = num_document + num_topic
    x_3 = num_document + num_topic + num_entity
    return [torch.arange(0, x_1), torch.arange(x_1, x_2), torch.arange(x_2, x_3)]


def fast_load(path=fast_snippet_path):
    """
    载入Snippets数据（快速模式，数据占空间较大）。
    输出为主题特征、文档特征、实体特征、主题-主题关系、主题-文档关系、主题-实体关系、文档-文档关系、文档-实体关系、实体-实体关系和类别信息。

    """
    _data = torch.load(path)
    _all_document_x = _data['features'][0]
    _all_topic_x = _data['features'][1]
    _all_entity_x = _data['features'][2]
    _all_adj_document = _data['adj'][0][0]
    _all_adj_document_topic = _data['adj'][0][1]
    _all_adj_document_entity = _data['adj'][0][2]
    _all_adj_topic = _data['adj'][1][1]
    _all_adj_topic_entity = _data['adj'][1][2]
    _all_adj_entity = _data['adj'][2][2]
    _all_labels = torch.argmax(_data['labels'], dim=1)
    return _all_document_x, _all_topic_x, _all_entity_x, _all_adj_document, _all_adj_document_topic, \
           _all_adj_document_entity, _all_adj_topic, _all_adj_topic_entity, _all_adj_entity, _all_labels


def load(path=snippet_path):
    """
    载入Snippets数据（正常模式，数据占空间小，已压缩）
    输出为主题特征、文档特征、实体特征、主题-主题关系、主题-文档关系、主题-实体关系、文档-文档关系、文档-实体关系、实体-实体关系和类别信息。

    """
    _data = np.load(path)
    _all_document_x = torch.tensor(_data['d_x'])
    _all_topic_x = torch.tensor(_data['t_x'])
    _all_entity_x = torch.tensor(_data['e_x'])
    _all_adj_document = torch.tensor(_data['adj_dd'])
    _all_adj_document_topic = torch.tensor(_data['adj_dt'])
    _all_adj_document_entity = torch.tensor(_data['adj_de'])
    _all_adj_topic = torch.tensor(_data['adj_tt'])
    _all_adj_topic_entity = torch.tensor(_data['adj_te'])
    _all_adj_entity = torch.tensor(_data['adj_ee'])
    _all_labels = torch.tensor(_data['d_y'])
    return _all_document_x, _all_topic_x, _all_entity_x, _all_adj_document, _all_adj_document_topic, \
           _all_adj_document_entity, _all_adj_topic, _all_adj_topic_entity, _all_adj_entity, _all_labels

# 加载数据集
if os.path.exists('data/snippets.pt'):
    all_document_x, all_topic_x, all_entity_x, all_adj_document, all_adj_document_topic, all_adj_document_entity, \
    all_adj_topic, all_adj_topic_entity, all_adj_entity, all_labels = fast_load()
else:
    all_document_x, all_topic_x, all_entity_x, all_adj_document, all_adj_document_topic, all_adj_document_entity, \
    all_adj_topic, all_adj_topic_entity, all_adj_entity, all_labels = load()

# 获取样本类别总数
num_classes = len(torch.unique(all_labels))

np_labels = all_labels.detach().numpy()

# 划分数据集为训练集、验证集和测试集
train_dict = []
for i in range(num_classes):
    class_indices = np.where(np_labels == i)[0]
    if len(class_indices) > num_per_class:
        train_dict.extend(np.random.choice(class_indices, num_per_class))
    else:
        train_dict.extend(class_indices)
train_dict = set(train_dict)

num_documents = len(all_document_x)
indices = set(np.arange(num_documents))
indices.difference_update(train_dict)

train_dict = list(train_dict)
np.random.shuffle(train_dict)

indices = list(indices)
np.random.shuffle(indices)

train_dict.extend(indices)
indices = train_dict

num_train = num_per_class * num_classes // 2
num_val = num_per_class * num_classes // 2

train_indices = indices[0:num_train]
val_indices = indices[num_train:num_train + num_val]
test_indices = indices[num_train + num_val::]

num_features = [all_document_x.shape[1], all_topic_x.shape[1], all_entity_x.shape[1]]

# 建立HGAT网络
net = HGraphConvolutionNet(num_features, num_hiddens, num_classes, num_layer, bias=False, device=train_device,
                           allow_attention=True, dropout=0.80)

# L2正则化（weight decay），不包含输出层
net_params = []
for idx, layer in enumerate(net.layers):
    if idx == len(net.layers) - 1:
        net_params.append({
            'params': layer.parameters(),
            'weight_decay': 0.0
        })
    else:
        net_params.append({
            'params': layer.parameters(),
            'weight_decay': weight_decay
        })
net_params.append({
    'params': net.attentions.parameters(),
    'weight_decay': weight_decay
})

# Adam优化
optimizer = optim.Adam(net_params, lr=learning_rate)

# 损失函数
loss_function = nn.CrossEntropyLoss()

# 如果已训练过模型，则加载已有模型
if os.path.exists(best_model_path):
    net.load_state_dict(torch.load(best_model_path, map_location=torch.device(train_device)))
else:
    # 获取训练数据和验证数据
    train_y = all_labels[train_indices].to(train_device)
    val_y = all_labels[val_indices].to(train_device)

    train_document_x = all_document_x[train_indices]
    train_adj_document = all_adj_document[train_indices, :][:, train_indices]
    train_adj_document_topic = all_adj_document_topic[train_indices, :]
    train_adj_document_entity = all_adj_document_entity[train_indices, :]

    val_document_x = all_document_x[val_indices]
    val_adj_document = all_adj_document[val_indices, :][:, val_indices]
    val_adj_document_topic = all_adj_document_topic[val_indices, :]
    val_adj_document_entity = all_adj_document_entity[val_indices, :]

    train_adj = block_diag(train_adj_document, train_adj_document_topic, train_adj_document_entity, all_adj_topic,
                           all_adj_topic_entity, all_adj_entity).to(train_device)
    train_adj = normalize_adj(train_adj, train_device)
    val_adj = block_diag(val_adj_document, val_adj_document_topic, val_adj_document_entity, all_adj_topic,
                         all_adj_topic_entity, all_adj_entity).to(train_device)
    val_adj = normalize_adj(val_adj, train_device)

    train_features = [train_document_x.to(train_device), all_topic_x.to(train_device), all_entity_x.to(train_device)]
    val_features = [val_document_x.to(train_device), all_topic_x.to(train_device), all_entity_x.to(train_device)]
    train_split = block_split(len(train_document_x), len(all_topic_x), len(all_entity_x))
    val_split = block_split(len(val_document_x), len(all_topic_x), len(all_entity_x))

    # 初始化绘图数据
    plot_x_data = []
    plot_train_acc_data = []
    plot_train_loss_data = []
    plot_val_acc_data = []
    plot_val_loss_data = []

    # 训练网络
    for epoch in range(max_epochs):
        net.train()
        optimizer.zero_grad()
        
        train_output = net(train_features, train_adj, train_split, normalize=False)[0:len(train_y)]
        train_loss = loss_function(train_output, train_y)
        train_pred_y = torch.argmax(train_output, dim=1)
        train_acc = torch.sum(train_y == train_pred_y).item() / len(train_y)

        loss = loss_function(train_output, train_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            net.eval()
            val_output = net(val_features, val_adj, val_split, normalize=False)[0:len(val_y)]
            val_loss = loss_function(val_output, val_y)
            val_pred_y = torch.argmax(val_output, dim=1)
            val_acc = torch.sum(val_y == val_pred_y).item() / len(val_y)
            print(
                f'Epoch={epoch}, train_loss={train_loss.item()}, train_acc={train_acc}, val_loss={val_loss.item()}, val_acc={val_acc}')

            plot_val_acc_data.append(val_acc)
            plot_train_acc_data.append(train_acc)
            plot_train_loss_data.append(train_loss.item())
            plot_val_loss_data.append(val_loss.item())
            plot_x_data.append(epoch)

            if (train_acc > 0.85 and val_acc > 0.85) or train_acc > 0.995:
                torch.save(net.state_dict(), best_model_path)
                print(f'Early stopping(train_acc): save network parameters.')
                break
    torch.save(net.state_dict(), best_model_path)

    plt.plot(plot_x_data, plot_train_acc_data, label='train')
    plt.plot(plot_x_data, plot_val_acc_data, label='val')
    plt.legend()
    plt.title('Training/Validation accuracy')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.savefig('figures/accuracy.pdf')
    plt.show()

    plt.plot(plot_x_data, plot_train_loss_data, label='train')
    plt.plot(plot_x_data, plot_val_loss_data, label='val')
    plt.legend()
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.title('Training/Validation loss')
    plt.savefig('figures/loss.pdf')
    plt.show()

# 测试集性能测试
gc.collect()
test_y = all_labels[test_indices].to(test_device)
test_document_x = all_document_x[test_indices]
test_adj_document = all_adj_document[test_indices, :][:, test_indices]
test_adj_document_topic = all_adj_document_topic[test_indices, :]
test_adj_document_entity = all_adj_document_entity[test_indices, :]
test_adj = block_diag(test_adj_document, test_adj_document_topic, test_adj_document_entity, all_adj_topic,
                      all_adj_topic_entity, all_adj_entity).to(test_device)

test_features = [test_document_x.to(test_device), all_topic_x.to(test_device), all_entity_x.to(test_device)]
test_split = block_split(len(test_document_x), len(all_topic_x), len(all_entity_x))
net.to(test_device)
net.device = test_device
for layer_attentions in net.attentions:
    for attention in layer_attentions:
        attention.device = test_device
test_output = net(test_features, test_adj, test_split)[0:len(test_y)]
test_loss = loss_function(test_output, test_y)
test_pred_y = torch.argmax(test_output, dim=1)
test_acc = torch.sum(test_y == test_pred_y).item() / len(test_y)
print(f'Test: test_loss={test_loss}, test_acc={test_acc}')
