#!/usr/bin/env python
# coding: utf-8

from pres_gpt2 import PresGPT2, GPTConfig 
from Dataset import PresidentDataset

import tiktoken
import pickle
import torch
from torch.utils.data import Dataset
from MyTrainer import MyTrainer
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import os

def load_train_objs():
    with open("pres_tokenizer.pkl", "rb") as f:
        pres_enc: tiktoken.Encoding = pickle.load(f)
        
    with open("train.pkl", "rb") as f:
        train = pickle.load(f)
        
    with open("validation.pkl", "rb") as f:
        validation = pickle.load(f)

    train_dataset = PresidentDataset(train)
    validation_dataset = PresidentDataset(validation)

    config: GPTConfig = GPTConfig(
        1024,
        len(pres_enc._mergeable_ranks) + len(pres_enc._special_tokens),
        12,
        12,
        768
    )

    model: PresGPT = PresGPT(config)
    
    param_dict = { name: params for name, params in model.parameters() }
    param_dict_grad = { name: params for name, params in param_dict.items() if params.requires_grad }
    
    optim_params = [
        { 'params': [p for _, p in param_dict_grad.items() if p.dim() >= 2], 'weight_decay': 0.1 },
        { 'params': [p for _, p in param_dict_grad.items() if p.dim() < 2], 'weight_decay': 0 }
    ]

    optimizer = torch.optim.AdamW(optim_params, betas=(0.9, 0.95), lr=3e-4)
    
    return train_dataset, validation_dataset, model, optimizer, pres_enc


def prepare_dataloader(dataset: Dataset, batch_size: int, distributed: bool):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=not distributed,
        sampler=None if not distributed else DistributedSampler(dataset),
    )

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)

    # nccl is NVIDIA collective communications lib, popular backend for communication btwn nvidia GPUs
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def multi_gpu(rank, world_size, total_epochs, save_every):
    ddp_setup(rank, world_size)
    
    train_dataset, validation_dataset, model, optimizer, _ = load_train_objs()
    train_data = prepare_dataloader(train_dataset, batch_size=32, distributed=True)
    validation_data = prepare_dataloader(validation_dataset, batch_size=32, distributed=True)
    
    trainer = MyTrainer(model, train_data, validation_data, optimizer, True, False, rank, save_every, "")
    trainer.train(total_epochs)
    
    destroy_process_group()
    

# start training job
if __name__ == '__main__':
    total_epochs = 50
    save_every = 2
    world_size = torch.cuda.device_count()
    print('world size is : ', world_size)
    mp.spawn(multi_gpu, args=(world_size, total_epochs, save_every,), nprocs=world_size) 
