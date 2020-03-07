import argparse

import torch
from torch import optim
import torch.nn as nn

from data_loader import DataLoader
import data_loader

import matplotlib.pyplot as plt

from simple_nmt.seq2seq import Seq2Seq
from simple_nmt.encoder import Encoder
from simple_nmt.decoder import Decoder
from simple_nmt.decoder import Generator
from simple_nmt.attention import Attention
from simple_nmt.rnnlm import LanguageModel

from hyperParams import HyperParams


def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    
    loader = DataLoader(
        config.train,
        config.valid,
        (config.lang[:17], config.lang[17:]),
        batch_size=config.lm_batch_size,
        max_length=config.max_length
    )

'''
현재 코드에서는 lm_trainer만 사용
(따라서 train하는 과정에서는 Seq2Seq모델 정의 부분은 없어도 됨)
'''

    from simple_nmt.lm_trainer import LanguageModelTrainer as LMTrainer

    language_models = [
        LanguageModel(
            len(loader.tgt.vocab),
            config.word_vec_size,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        ),
        LanguageModel(
            len(loader.src.vocab),
            config.word_vec_size,
            config.hidden_size,
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        )
    ]

    models = [
        Seq2Seq(
            len(loader.src.vocab),
            config.word_vec_size,
            config.hidden_size,
            len(loader.tgt.vocab),
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        ),
        Seq2Seq(
            len(loader.tgt.vocab),
            config.word_vec_size,
            config.hidden_size,
            len(loader.src.vocab),
            n_layers=config.n_layers,
            dropout_p=config.dropout,
        )
    ]

    loss_weights = [
        torch.ones(len(loader.tgt.vocab)),
        torch.ones(len(loader.src.vocab)),
    ]
    loss_weights[0][data_loader.PAD] = .0
    loss_weights[1][data_loader.PAD] = .0

    crits = [
        nn.NLLLoss(weight=loss_weights[0], reduction='none'),
        nn.NLLLoss(weight=loss_weights[1], reduction='none'),
    ]

    print(language_models)
    print(models)
    print(crits)

    if model_weight is not None:
        for model, w in zip(models + language_models, model_weight):
            model.load_state_dict(w)

    # if config.gpu_id >= 0:
    #     for lm, seq2seq, crit in zip(language_models, models, crits):
    #         lm.cuda(config.gpu_id)
    #         seq2seq.cuda(config.gpu_id)
    #         crit.cuda(config.gpu_id)
    
    for lm, crit in zip(language_models, crits):
        optimizer = optim.Adam(lm.parameters())
        lm_trainer = LMTrainer(config)

        lm_trainer.train(
            lm, crit, optimizer,
            train_loader=loader.train_iter,
            valid_loader=loader.valid_iter,
            src_vocab=loader.src.vocab if lm.vocab_size == len(loader.src.vocab) else None,
            tgt_vocab=loader.tgt.vocab if lm.vocab_size == len(loader.tgt.vocab) else None,
            n_epochs=config.lm_n_epochs,
        )
        # print(losslist)

    
if __name__ == '__main__':
    config = HyperParams()
    main(config)
