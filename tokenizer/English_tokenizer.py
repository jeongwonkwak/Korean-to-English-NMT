import os
import sys
from torchtext import data, datasets
import sentencepiece as spm
import pandas as pd
import MeCab




def English_tokenizer():
    
    df = pd.read_csv("test.csv")
    ENG_data = df['English']
    
    f = open("eng.txt", "w", encoding = 'utf-8')
    for row in ENG_data[:100000]:
        f.write(row)
        f.write('\n')
    f.close()
    
    spm.SentencePieceTrainer.Train('--input=eng.txt \
                               --model_prefix=english_tok \
                               --vocab_size=100000\
                               --hard_vocab_limit=false')
    
    sp = spm.SentencePieceProcessor()
    sp.Load('english_tok.model')
    
    return lambda x : sp.EncodeAsPieces(x)
