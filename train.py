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
txt_files = [data_dir+file_name for file_name in os.listdir(data_dir)][::10]


if __name__ == '__main__':

    writer = SummaryWriter('runs/experiment-1')

    model = LanguageModel(n_vocab=10000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    criterion = nn.CrossEntropyLoss()

    dummy_input = torch.LongTensor([[1,2,3,4]]).to(device)
    writer.add_graph(model, dummy_input)

    global_step = 0
    for epoch in range(10):
        data_loader_iter = tqdm(TextDataLoaderIterator(txt_files, batch_size=64, block_len=64), position=0)
        for file_name, data_loader in data_loader_iter:
            pbar = tqdm(data_loader, desc=file_name, leave=False, position=1)
            for mlm_seq, original_seq in pbar:
                mask = torch.zeros_like(mlm_seq)
                mask[mlm_seq == original_seq] = 1.
                mlm_seq, mask, original_seq = mlm_seq.to(device), mask.to(device), original_seq.to(device)

                optimizer.zero_grad()
                output, *_ = model(mlm_seq, src_mask=mask)
                loss = sum([criterion(output[:, i, :], original_seq[:, i]) for i in range(output.size(0))]) / output.size(0)

                loss.backward()
                optimizer.step()

                global_step += 1
                if global_step % 100 == 0:
                    lr_scheduler.step()

                writer.add_scalar('Loss/Train', loss.item(), global_step)
                pbar.set_postfix({'loss': loss.item()})

        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            }, f'models/lm/epoch-{epoch+1}-gs-{global_step}.pth')
