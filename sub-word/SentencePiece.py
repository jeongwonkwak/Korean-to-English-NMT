import sentencepiece as spm
import pandas as pd

data = pd.read_csv("test.csv")
KOR_data = data['Korean']

f = open("C:/Users/USER/Capstone/data_kor.txt", "w")
for row in KOR_data[:10]:
    f.write(row)
f.close()

spm.SentencePieceTrainer.Train('--input=C:/Users/USER/Capstone/data_kor.txt \
                               --model_prefix=m \
                               --vocab_size=1000 \
                               --hard_vocab_limit=false')

sp = spm.SentencePieceProcessor()
sp.Load('m.model')


# input setence 
sentence = "안녕하세요. 감사합니다."


# 1. text -> subword
print("1. text -> subword", "\n")
a = sp.EncodeAsPieces(sentence)


# 2. text -> subword id
print("2. text -> subword id \n")
print(sp.EncodeAsIds(sentence))


# 3. subword id -> text
print("3. subword id -> text \n")
print(sp.DecodeIds(sp.EncodeAsIds(sentence)))


# 4. subword -> text
print("4. subword -> text \n")
print(sp.DecodePieces(sp.EncodeAsPieces(sentence)))


# 5. subword size
print("5. subword size \n")
print(sp.GetPieceSize())


 
