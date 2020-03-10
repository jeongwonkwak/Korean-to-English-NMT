import torch
import torch.nn as nn
import torch.optim as optim
import easydict

def HyperParams():

    hparams = easydict.EasyDict({

    "model_fn": 'revise.model',
    "train": 'C:/Users/USER/Capstone/', 
    "valid": 'C:/Users/USER/Capstone/',
    "lang": 'Korean_sample.csvEnglish_sample.csv',
    "gpu_id": 0,
        
    "batch_size": 32,
    "n_epochs": 15,
    "verbose": 2,
    "init_epoch": 1,
    "max_length": 80,
    "dropout": 0.2,

    "word_vec_size": 512,
    "hidden_size": 768,
    "n_layers": 4,
    "max_grad_norm": 5.0,

    "use_adam": True,
    "lr": 1.0,
    "Ir_step": 1,
    "Ir_gamma": 0.5,
    "lr_decay_start": 10,
    "use_noam_decay": True,
    "lr_n_warmup_steps": 48000,

    "lm_n_epochs": 2,
    "lm_batch_size": 5,

    "use_transformer": False,
    "n_splits": 8,
    "use_cuda": 1,

    "dsl": True,
    "dsl_n_epochs": 2,
    "dsl_lambda": 1e-3
    })
    
    return hparams


def HyperParams_translate():

    hparams_translate = easydict.EasyDict({

    "model": 'revise.17.1.37-3.94.1.13-3.09.1.20-3.33.1.02-2.78.model',
    "train": 'C:/Users/USER/Capstone/', 
    "valid": 'C:/Users/USER/Capstone/',
    "lang": 'Korean_sample.csvEnglish_sample.csv',
    "gpu_id": 0,
        
    "batch_size": 128,
    "max_length": 255,
    "n_best": 1,

    "beam_size": 5,
    "length_penalty": 1.2
    })
    
    return hparams_translate
