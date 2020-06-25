import os
from tqdm.contrib.concurrent import thread_map


def utf8enc(file_name):
    with open(file_name, 'rb') as f:
        text = f.read().decode('utf-8','ignore').encode("utf-8")
    with open(file_name, 'wb') as f:
        f.write(text)


def paragraph2line(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        text = [string.replace('\n', ' ') for string in f.read().split('\n\n')]
    with open(file_name, 'w') as f:
        f.write('\n\n'.join(text))


def proprocess(file_name):
    utf8enc(file_name)
    paragraph2line(file_name)


if __name__ == '__main__':
    data_dir = 'data/Gutenberg/txt/'
    txt_files = [data_dir+file_name for file_name in os.listdir(data_dir)]
    thread_map(proprocess, txt_files, max_workers=os.cpu_count())