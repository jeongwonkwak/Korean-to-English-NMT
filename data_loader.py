import os
import sys
from torchtext import data, datasets
import sentencepiece as spm
import pandas as pd
import MeCab


PAD = 1
BOS = 2
EOS = 3


class DataLoader():
    
    def __init__(self, train_fn = None, 
                    valid_fn = None, 
                    exts = None,
                    batch_size = 64, 
                    device = 'cpu', 
                    max_vocab = 99999999,    
                    max_length = 255, 
                    fix_length = None, 
                    use_bos = True, 
                    use_eos = True, 
                    shuffle = True
                    ):

        """
        * sequential : 데이터의 유형이 연속형 데이터인지(False면 토큰화가 적용x)
        * use_vocab : Vocab 사용 여부(False면 필드의 데이터는 이미 숫자여야함)
        * batch_first : 배치 수가 먼저 텐서를 생성할지 여부
        * include_lengths : 패딩 된 미니 배치의 튜플과 각 예제의 길이를 포함하는 목록 
                            또는 패딩 된 미니 배치를 반환할지 여부(default: False)
        * fix_length : 모든 문장이 채워지는 고정 길이, 유연한 sequence의 경우 None
        * init_token : 모든 문장 앞에 추가되는 토큰
        * eos_token : 모든 문장 뒤에 추가되는 토큰
        """
        super(DataLoader, self).__init__()
        
        # 한국어(source) 데이터의 틀
        self.src = data.Field(sequential = True,
                                use_vocab = True, 
                                batch_first = True, 
                                include_lengths = True, 
                                fix_length = fix_length, 
                                init_token = None, 
                                eos_token = None,
                                tokenize = Korean_tokenizer()
                                )
        super(DataLoader, self).__init__()
        
        # 영어(target) 데이터의 틀
        self.tgt = data.Field(sequential = True, 
                                use_vocab = True, 
                                batch_first = True, 
                                include_lengths = True, 
                                fix_length = fix_length, 
                                init_token = '<BOS>' if use_bos else None, 
                                eos_token = '<EOS>' if use_eos else None,
                                tokenize = English_tokenizer()
                                )
        
        
        # train, test(valid) 데이터 틀(앞서 생성한 src, tgt 각각)
        train = TranslationDataset(path = train_fn, exts = exts,
                                        fields = [('src', self.src), ('tgt', self.tgt)], 
                                        max_length = max_length
                                        )
        valid = TranslationDataset(path = valid_fn, exts = exts,
                                        fields = [('src', self.src), ('tgt', self.tgt)], 
                                        max_length = max_length
                                        )
        
        # 각각의 epoch에 대해 새로 섞인 batch를 생성하면서 반복
        # train, test(valid)를 각각 나눴으니 아래 작업은 진행해도 무방할 듯
        self.train_iter = data.BucketIterator(train, 
                                                batch_size = batch_size, 
                                                shuffle = shuffle, 
                                                sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)), 
                                                sort_within_batch = True
                                                )

        self.valid_iter = data.BucketIterator(valid, 
                                                batch_size = batch_size, 
                                                shuffle = False, 
                                                sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)), 
                                                sort_within_batch = True
                                                )
        
        self.src.build_vocab(train, max_size = max_vocab)
        self.tgt.build_vocab(train, max_size = max_vocab)

    def load_vocab(self, src_vocab, tgt_vocab):
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab
        
        
        
class TranslationDataset(data.Dataset):

    def sort_key(ex):  # 음수와 양수 모두 가능
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, max_length=None, **kwargs):
        
        """
        * path : 두 언어의 데이터 파일 경로
        * exts : 각 언어의 경로 확장을 포함하는 튜플
        * fields : 각 언어의 데이터에 사용될 필드를 포함하는 튜플
        * **kwargs : 생성자에 전달 
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
        
        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if max_length and max_length < max(len(src_line.split()), len(trg_line.split())):
                    continue
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(TranslationDataset, self).__init__(examples, fields, **kwargs)
 
'''        
if __name__ == '__main__':

    """
    argv1,2 : src.csv와 tgt.csv파일이 있는 공통 경로
    (argv3, argv4) : 확장자를 포함한 각 파일 이름
    """
    loader = DataLoader('C:/Users/USER/Capstone/','C:/Users/USER/Capstone/' , ('Korean_sample.csv','English_sample.csv'),
                        shuffle = False, 
                        batch_size = 8
                        )
    
    
    print(len(loader.src.vocab))
    print(len(loader.tgt.vocab))
    
    for batch_index, batch in enumerate(loader.train_iter):
        print(batch_index)
        print(batch.src)
        print(batch.tgt)
        
        if batch_index > 1:
            break
'''
