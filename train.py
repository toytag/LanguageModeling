import os
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import TextDataLoaderIterator
from model import LanguageModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_dir = 'data/Gutenberg/txt/'
txt_files = [data_dir + file_name for file_name in os.listdir(data_dir)]


if __name__ == '__main__':

    checkpoint = torch.load('models/lm/latest.pth')

    model = LanguageModel(n_vocab=10000).to(device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, threshold=0.01, min_lr=1e-6)
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter('runs/')
    dummy_input = torch.ones(()).to(device)
    writer.add_graph(model, dummy_input)

    # global_step = checkpoint['global_step']
    global_step = 0

    for epoch in range(10):
        pbar = tqdm(TextDataLoaderIterator(txt_files, batch_size=16, block_len=16)
        for data_loader in data_loader_iter:
            for mlm_seq, original_seq in data_loader:
                mask = torch.zeros_like(mlm_seq)
                mask[mlm_seq == original_seq] = 1.
                mlm_seq, mask, original_seq = mlm_seq.to(device), mask.to(device), original_seq.to(device)

                optimizer.zero_grad()
                output, *_ = model(mlm_seq, src_mask=mask)
                loss = sum([criterion(output[:, i, :], original_seq[:, i]) for i in range(output.size(0))]) / output.size(0)

                loss.backward()
                optimizer.step()

                lr_scheduler.step(loss)
                global_step += 1

                writer.add_scalar('Loss', loss.item(), global_step)
                writer.add_scalar('Lr', optimizer.get_lr(), global_step)

            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.get_lr()})

        torch.save({
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }, 'models/lm/latest.pth')
