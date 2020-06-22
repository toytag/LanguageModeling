import os, time
from multiprocessing import Pool, Manager

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import ByteLevelBPETokenizer
from tqdm import tqdm


tokenizer = ByteLevelBPETokenizer(
    'models/tokenizer/vocab.json',
    'models/tokenizer/merges.txt',
    lowercase=True
)


class TextDataset(Dataset):
    def __init__(self, txt_file, bptt_len=16):
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            textlines = f.readlines()
        self.dataset = []
        for line in textlines:
            self.dataset += tokenizer.encode(line, add_special_tokens=False).ids
        self.bptt_len = bptt_len

    def __len__(self):
        return len(self.dataset) - self.bptt_len - 1

    def __getitem__(self, idx):
        seq = torch.tensor(self.dataset[idx:idx+self.bptt_len], dtype=torch.long)
        label = torch.tensor(self.dataset[idx+self.bptt_len], dtype=torch.long)
        return seq, label


def _enqueue_dataloader(txt_file, batch_size, bptt_len, shuffle, pin_memory, num_workers, q):
    # enqueue (file_name, DataLoader)
    q.put((txt_file, DataLoader(TextDataset(txt_file, bptt_len), batch_size, 
        shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)))


class TextDataLoaderIterator:
    def __init__(self, txt_files, batch_size=32, bptt_len=16, shuffle=True,
                 pin_memory=False, num_workers=os.cpu_count(), prefetch_limit=10):
        self.m = Manager()
        self.p = Pool(num_workers)
        self.q = self.m.Queue(prefetch_limit)

        for txt_file in txt_files:
            self.p.apply_async(_enqueue_dataloader, 
                (txt_file, batch_size, bptt_len, shuffle, pin_memory, num_workers, self.q))

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
    data_dir = '../data/Gutenberg/txt/'
    txt_files = [data_dir+file_name for file_name in os.listdir(data_dir)][-10:]

    dataloaders = TextDataLoaderIterator(txt_files, 512, 64)
    for dataloader in dataloaders:
        # for seq, label in tqdm(dataloader):
        #     # print(seq.size(), label.size())
        print(dataloader)
        
