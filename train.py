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

from hyperparams import HyperParams


def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    if config.dsl:
        
        loader = DataLoader(
            config.train,
            config.valid,
            (config.lang[:17], config.lang[17:]),
            batch_size=config.lm_batch_size,
            device=config.gpu_id,
            max_length=config.max_length,
            dsl=config.dsl,
        )

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
            ),
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
            ),
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

        losslist_src, losslist_tgt = [], []

        for index, lm_crit in enumerate(zip(language_models, crits)):
            lm, crit = lm_crit
            optimizer = optim.Adam(lm.parameters())
            lm_trainer = LMTrainer(config)

            
            model_re, losslist = lm_trainer.train(
                lm, crit, optimizer,
                train_loader=loader.train_iter,
                valid_loader=loader.valid_iter,
                src_vocab=loader.src.vocab if lm.vocab_size == len(loader.src.vocab) else None,
                tgt_vocab=loader.tgt.vocab if lm.vocab_size == len(loader.tgt.vocab) else None,
                n_epochs=config.lm_n_epochs,
            )


            if index == 0:
                losslist_src = losslist 
            else:
                losslist_tgt = losslist

        # print(losslist_src)
        # print(losslist_tgt)
            
        # plt.plot(losslist_src, label = "Korean")
        # plt.plot(losslist_tgt, label = "English")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        loader = DataLoader(
            config.train,
            config.valid,
            (config.lang[:17], config.lang[17:]),
            batch_size=config.batch_size,
            device=config.gpu_id,
            max_length=config.max_length,
            dsl=config.dsl
        )

        from simple_nmt.dual_trainer import DualSupervisedTrainer as DSLTrainer
        dsl_trainer = DSLTrainer(config)

        optimizers = [
            optim.Adam(models[0].parameters()),
            optim.Adam(models[1].parameters()),
        ]

        if opt_weight is not None:
            for opt, w in zip(optimizers, opt_weight):
                opt.load_state_dict(w)

        dsl_trainer.train(
            models,
            language_models,
            crits,
            optimizers,
            train_loader=loader.train_iter,
            valid_loader=loader.valid_iter,
            vocabs=[loader.src.vocab, loader.tgt.vocab],
            n_epochs=config.n_epochs + config.dsl_n_epochs,
            lr_schedulers=None,
        )
    

if __name__ == '__main__':
    config = HyperParams()
    main(config)
