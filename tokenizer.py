import os
from tokenizers import ByteLevelBPETokenizer


if __name__ == '__main__':
    data_dir = 'data/Gutenberg/txt/'
    txt_files = [data_dir + file_name for file_name in os.listdir(data_dir)]

    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    tokenizer.train(txt_files, vocab_size=10000, min_frequency=10, special_tokens=[
        "<s>", "</s>", "<pad>", "<unk>", "<eos>", "<mask>",
    ])
    tokenizer.save('models/tokenizer/')

    print('vocab_size', tokenizer.get_vocab_size())