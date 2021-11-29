
import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='./data/', type=str, help="数据保存路径")
    parser.add_argument("--save_dir", default='./outputs/', type=str, help="结果保存路径")
    parser.add_argument("--tokenizer_path", default='./model_weight/NEZHA/vocab.txt', type=str, help='词表的位置')

    parser.add_argument("--aug_data", default=False, type=bool, help="是否进行数据增强")
    parser.add_argument("--use_adv", default='pgd', type=str, help="使用对抗训练")
    parser.add_argument("--joint_len_limit", default=32, type=int, help="")
    parser.add_argument("--len_limit", default=16, type=int, help="单个输入序列的最大长度")

    parser.add_argument("--epochs", default=5, type=int, help='将数据训练几轮')
    parser.add_argument("--learning_rate", default=1.5e-5, type=float, help='学习率的大小')
    parser.add_argument("--weight_decay", default=1e-3, type=float, help='')

    parser.add_argument('--train_batch_size', default=256, type=int, help='训练的批次大小')
    parser.add_argument('--eval_batch_size', default=256, type=int, help='验证的批次大小')

    parser.add_argument('--task_type', default='a', type=str, help='是否将两种数据分开，可设置为a, b, ab')

    parser.add_argument('--use_scheduler', default=True, type=bool, help='学习率动态变化 ')

    return parser.parse_args()


