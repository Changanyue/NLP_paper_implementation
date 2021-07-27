import numpy as np
from numpy.random import shuffle
import pandas as pd

# ===================preprocessing:==============================
def load_dataset(name):
    assert name in ['20NG','AG','DBP','FDU','THU'], "name only supports '20NG','AG','DBP','FDU','THU', but your input is %s"%name

    if name == '20NG':
        num_classes = 20
        df1 = pd.read_csv('datasets/20NG/20ng-train-all-terms.csv')
        df2 = pd.read_csv('datasets/20NG/20ng-test-all-terms.csv')
        df = pd.concat([df1,df2])
        comp_group = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                      'comp.windows.x']
        rec_group = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
        talk_group = ['talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
        sci_group = ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space']
        other_group = ['alt.atheism', 'misc.forsale', 'soc.religion.christian']
        label_groups = [comp_group,rec_group,talk_group]
        return df, num_classes,label_groups
    if name == 'AG':
        num_classes = 4  # DBP:14 AG:4
        df1 = pd.read_csv('../datasets/AG_news/train.csv', header=None, index_col=None)
        df2 = pd.read_csv('../datasets/AG_news/test.csv', header=None, index_col=None)
        df1.columns = ['label', 'title', 'content']
        df2.columns = ['label', 'title', 'content']
        df = pd.concat([df1, df2])
        label_groups = []
        return df,num_classes,label_groups
    if name == 'DBP':
        num_classes = 14
        df1 = pd.read_csv('../datasets/DBPedia/train.csv', header=None, index_col=None)
        df2 = pd.read_csv('../datasets/DBPedia/test.csv', header=None, index_col=None)
        df1.columns = ['label', 'title', 'content']
        df2.columns = ['label', 'title', 'content']
        df = pd.concat([df1, df2])
        label_groups = []
        return df,num_classes,label_groups
    if name == 'FDU':
        num_classes = 20
        df = pd.read_csv('../datasets/fudan_news.csv')
        label_groups = []
        return df, num_classes,label_groups
    if name == 'THU':
        num_classes = 13
        df = pd.read_csv('F:/label_confusion_learning/datasets/thucnews_subset.csv')
        label_groups = []
        return df, num_classes,label_groups

def create_asy_noise_labels(y, label_groups, label2idx, rate):
    np.random.seed(0)
    """
    y: the list of label index
    label_groups: each group is a list of label names to exchange within groups
    rate: noise rate, in this mode, each selected label will change to another random label in the same group
    """
    y_orig = list(y)[:]
    y = np.array(y)
    count = 0
    for labels in label_groups:
        label_ids = [label2idx[l] for l in labels]
        # find out the indexes of your target labels to add noise
        indexes = [i for i in range(len(y)) if y[i] in label_ids]
        shuffle(indexes)
        partial_indexes = indexes[:int(rate * len(indexes))]
        count += len(partial_indexes)
        # find out the indexes of your target labels to add noise
        for idx in partial_indexes:
            if y[idx] in label_ids:
                # randomly change to another label in the same group
                other_label_ids = label_ids[:]
                other_label_ids.remove(y[idx])
                y[idx] = np.random.choice(other_label_ids)
    errors = len(y) - list(np.array(y_orig) - np.array(y)).count(0)
    shuffle_rate = count / len(y)
    error_rate = errors / len(y)
    return shuffle_rate, error_rate, y

