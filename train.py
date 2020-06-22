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
txt_files = [data_dir+file_name for file_name in os.listdir(data_dir)][::100]


if __name__ == '__main__':

    writer = SummaryWriter('runs/experiment-1')

    model = LanguageModel(n_vocab=30000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    criterion = nn.CrossEntropyLoss()

    dummy_input = torch.LongTensor([[1,2,3,4]]).to(device)
    writer.add_graph(model, dummy_input)

    global_step = 0

    for epoch in range(10):
        epoch_loss = 0
        total_num_loss = 0

        data_loader_iter = tqdm(TextDataLoaderIterator(txt_files, batch_size=512, bptt_len=16), position=0)
        for file_name, data_loader in data_loader_iter:
            pbar = tqdm(data_loader, desc=file_name, leave=False, position=1)
            for seq, label in pbar:
                seq, label = seq.to(device), label.to(device)

                optimizer.zero_grad()
                output, *_ = model(seq)
                loss = criterion(output[:, -1, :], label)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_num_loss += 1
                global_step += 1

                if global_step % 1000 == 0:
                    lr_scheduler.step()

                writer.add_scalar('Loss/Train', loss.item(), total_num_loss, global_step)
                pbar.set_postfix({'loss': loss.item()})

        epoch_loss /= total_num_loss
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': epoch_loss,
            }, f'models/lm/epoch-{epoch+1}-loss-{epoch_loss:.2f}.pth')
