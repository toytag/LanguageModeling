import os, time
from multiprocessing import Pool, Manager

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import WordTokenizer


_tokenizer = WordTokenizer('models/tokenizer/tokenizer.json')


class TextDataset(Dataset):
    def __init__(self, txt_file, block_len=64, mlm_percentage=0.15):
        with open(txt_file, 'r', encoding='utf-8') as f:
            textlines = f.readlines()
        self.examples = []
        self.block_len = block_len
        for line in textlines:
            new_tokens = _tokenizer.encode(line)
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
        seq = torch.LongTensor(seq[i:i + self.block_len])
        mask_idx = np.random.randint(0, self.block_len, size=np.int(self.block_len * self.mlm_percentage))
        mask = torch.ones_like(seq)
        mask[mask_idx] = 0

        return seq, mask


def _enqueue_dataloader(txt_file, batch_size, block_len, shuffle, pin_memory, num_workers, q):
    # enqueue (file_name, DataLoader)
    q.put(DataLoader(TextDataset(txt_file, block_len), batch_size, 
        shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers))


class TextDataLoaderIterator:
    def __init__(self, txt_files, batch_size=64, block_len=64, shuffle=True,
                 pin_memory=True, num_workers=os.cpu_count(), prefetch_limit=10):
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
            self.p.close()
            raise StopIteration
        self.idx += 1
        return self.q.get()


from tqdm import tqdm
if __name__ == '__main__':
    data_dir = 'data/Gutenberg/txt/'
    txt_files = [data_dir+file_name for file_name in os.listdir(data_dir)][:2]

    dataloader_iter = TextDataLoaderIterator(txt_files, 64, 64)
    dataloader = next(iter(dataloader_iter))
    seq, mask = next(iter(dataloader))
    assert seq.shape == mask.shape