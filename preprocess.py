import os
from tqdm.contrib.concurrent import thread_map


def utf8enc(file_name):
    with open(file_name, 'rb') as f:
        text = f.read().decode('utf-8','ignore').encode("utf-8")
    with open(file_name, 'wb') as f:
        f.write(text)


if __name__ == '__main__':
    data_dir = '../data/Gutenberg/txt/'
    txt_files = [data_dir+file_name for file_name in os.listdir(data_dir)]
    thread_map(utf8enc, txt_files, max_workers=os.cpu_count())