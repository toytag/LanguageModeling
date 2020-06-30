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


data_dir = 'data/Gutenberg/split/'
txt_files = [data_dir + file_name for file_name in os.listdir(data_dir)]


if __name__ == '__main__':

    # checkpoint = torch.load('models/lm/latest.pth')

    model = LanguageModel(n_vocab=10000).to(device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=30, threshold=0.001, min_lr=1e-6)
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter('runs/')
    dummy_input = torch.LongTensor([[1]]).to(device)
    writer.add_graph(model, dummy_input)

    # global_step = checkpoint['global_step']
    global_step = 0

    for epoch in range(10):
        pbar = tqdm(TextDataLoaderIterator(txt_files, batch_size=16, block_len=16))
        for data_loader in pbar:
            for seq, mask in data_loader:
                seq, mask = seq.to(device), mask.to(device)

                output, *_ = model(seq.masked_fill(mask==0, LanguageModel.mask_idx), src_mask=mask)
                loss = criterion(output[mask==0], seq[mask==0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                lr_scheduler.step(loss)
                global_step += 1

                writer.add_scalar('loss', loss.item(), global_step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

            pbar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        torch.save({
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }, 'models/lm/latest.pth')
