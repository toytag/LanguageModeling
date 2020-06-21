import os
from tokenizers import BertWordPieceTokenizer


if __name__ == '__main__':
    data_dir = '../data/Gutenberg/txt/'
    txt_files = [data_dir + file_name for file_name in os.listdir(data_dir)]

    tokenizer = BertWordPieceTokenizer(lowercase=True)
    tokenizer.train(txt_files, vocab_size=30000, min_frequency=10)
    tokenizer.save('models/tokenizer/')

    print('vocab_size', tokenizer.get_vocab_size())