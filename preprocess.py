import os, shutil
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map


data_dir = 'data/Gutenberg/txt/'
txt_files = [(i, data_dir + file_name) for i, file_name in enumerate(os.listdir(data_dir))]
split_file_dir = 'data/Gutenberg/split/'


def proprocess(args):
    idx, file_name = args
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        text = [string.replace('\n', ' ') for string in f.read().split('\n\n')]
    with open(split_file_dir + f'f{idx%100:02d}.txt', 'a') as f:
        f.write('\n\n'.join(text) + '\n')


if __name__ == '__main__':
    if os.path.exists(split_file_dir):
        shutil.rmtree(split_file_dir)
    os.mkdir(split_file_dir)

    thread_map(proprocess, txt_files, max_workers=os.cpu_count())