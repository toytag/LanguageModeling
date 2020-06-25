import os, time
from multiprocessing import Pool, Manager

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import ByteLevelBPETokenizer
from tqdm import tqdm
import numpy as np


tokenizer = ByteLevelBPETokenizer(
    'models/tokenizer/vocab.json',
    'models/tokenizer/merges.txt',
    lowercase=True
)


class TextDataset(Dataset):
    def __init__(self, txt_file, block_len=64, mlm_percentage=0.15):
        with open(txt_file, 'r', encoding='utf-8') as f:
            textlines = f.readlines()
        self.examples = []
        self.block_len = block_len
        for line in textlines:
            new_tokens = tokenizer.encode(line).ids
            if len(new_tokens) <= self.block_len:
                continue
            self.examples.append(new_tokens)
        self.mlm_percentage = mlm_percentage

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        seq = self.examples[idx]
        seq_len = len(seq)
        i = np.random.randint(0, seq_len - self.block_len)
        seq = seq[i:i + self.block_len]
        mlm_i = np.random.randint(0, self.block_len, size=np.int(self.block_len * self.mlm_percentage))
        mlm_seq = np.array(seq)
        mlm_seq[mlm_i] = tokenizer.token_to_id('<mask>')
        mlm_seq = torch.tensor(mlm_seq, dtype=torch.long)
        original_seq = torch.tensor(seq, dtype=torch.long)

        return mlm_seq, original_seq


def _enqueue_dataloader(txt_file, batch_size, block_len, shuffle, pin_memory, num_workers, q):
    # enqueue (file_name, DataLoader)
    q.put((txt_file, DataLoader(TextDataset(txt_file, block_len), batch_size, 
        shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)))


class TextDataLoaderIterator:
    def __init__(self, txt_files, batch_size=64, block_len=64, shuffle=True,
                 pin_memory=False, num_workers=os.cpu_count(), prefetch_limit=10):
        self.m = Manager()
        self.p = Pool(num_workers)
        self.q = self.m.Queue(prefetch_limit)

        for txt_file in txt_files:
            self.p.apply_async(_enqueue_dataloader, 
                (txt_file, batch_size, block_len, shuffle, pin_memory, num_workers, self.q))

        self.idx = 0
        self.end_idx = len(txt_files)

    def __len__(self):
        return self.end_idx

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx == self.end_idx:
            raise StopIteration
        self.idx += 1
        return self.q.get()


from tqdm import tqdm
if __name__ == '__main__':
    data_dir = 'data/Gutenberg/txt/'
    txt_files = [data_dir+file_name for file_name in os.listdir(data_dir)][:2]

    dataloaders = TextDataLoaderIterator(txt_files, 64, 64)
    for txt_file, dataloader in dataloaders:
        for mlm_seq, original_seq in tqdm(dataloader):
            mask = torch.zeros_like(mlm_seq)
            mask[mlm_seq == original_seq] = 1.
            print(mask)
            # print(mlm_seq, original_seq)
            # pass
        