from torchtext.datasets import SST
from torchtext.data import Field, BucketIterator
from torchtext.vocab import GloVe
TEXT = Field(lower=True,fix_length=200,batch_first=True)
LABEL = Field(sequential=False)
train,valid,test = SST.splits(TEXT,LABEL)
TEXT.build_vocab(train,vectors=GloVe(name='6B',dim=100),max_size=20000,min_freq=10)
LABEL.build_vocab(train)
train_iter, valid_iter, test_iter = BucketIterator.splits((train,valid,test),batch_size=16)