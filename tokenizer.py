import re, json
from collections import Counter


class WordPieceTokenizer:
    def __init__(
            self,
            vocab_file=None,
            vocab_size_limit=10000, 
            min_freq=10, 
            lowercase=True, 
            special_tokens=[]
        ):

        pattern = r'[\w\']+|[,<.>/?;:\'\"\[\{\]\}\\\|\-_=\+`~!@#$%^&*\(\)]'

        if vocab_file:
            with open(vocab_file, 'r') as f:
                config = json.load(f)
            vocab_size_limit = config['vocab_size_limit']
            min_freq = config['min_freq']
            lowercase = config['lowercase']
            special_tokens = config['special_tokens']
            
        self.p = re.compile(r''.join([s + r'|' for s in special_tokens]) + pattern)
            
        self.vocab_size_limit = vocab_size_limit
        self.min_freq = min_freq
        self.lowercase = lowercase
        self.special_tokens = special_tokens

        self.vocab = Counter() if not vocab_file else Counter(config['vocab'])

        self.vocab_size = 0
        self.token2id = {}
        self.token2id_updated = False
        self.id2token = {}
        self.id2token_updated = False


    def build_vocab(self, corpus):
        if self.lowercase:
            corpus = corpus.lower()
        
        tokens = self.p.findall(corpus)
        self.vocab += Counter(tokens)

        self.token2id_updated = False
        self.id2token_updated = False


    def _update_token2id(self):
        self.token2id = {special_token: i for i, special_token in enumerate(self.special_tokens)}
        
        for i, (vocab, freq) in enumerate(self.vocab.most_common()):
            i = i + len(self.special_tokens)
            if freq >= self.min_freq and i < self.vocab_size_limit:
                self.token2id[vocab] = i

        self.vocab_size = len(self.token2id.keys())
        self.token2id_updated = True


    def _update_id2token(self):
        self.id2token = {i: special_token for i, special_token in enumerate(self.special_tokens)}

        for i, (vocab, freq) in enumerate(self.vocab.most_common()):
            i = i + len(self.special_tokens)
            if freq >= self.min_freq and i < self.vocab_size_limit:
                self.id2token[i] = vocab
                
        self.vocab_size = len(self.id2token.keys())
        self.id2token_updated = True


    def encode(self, text):
        if not self.token2id_updated:
            self._update_token2id()
        if self.lowercase:
            text = text.lower()
        return [self.token2id[token] for token in self.p.findall(text)]


    def decode(self, ids):
        if not self.id2token_updated:
            self._update_id2token()
        return ' '.join([self.id2token[id_] for id_ in ids])

    
    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                'vocab_size_limit': self.vocab_size_limit,
                'min_freq': self.min_freq,
                'lowercase': self.lowercase,
                'special_tokens': self.special_tokens,
                'vocab': self.vocab
            },f, indent=4)


import os
if __name__ == '__main__':

    data_dir = 'data/Gutenberg/split/'
    txt_files = [data_dir + file_name for file_name in os.listdir(data_dir)]

    tokenizer = WordPieceTokenizer(special_tokens=['<mask>'])

    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            text = f.read()
        tokenizer.build_vocab(text)
    
    print(tokenizer.decode(tokenizer.encode('what are you <mask>')))
    print('vocab size:', tokenizer.vocab_size)
    tokenizer.save('models/tokenizer/tokenizer.json')
    tokenizer = WordPieceTokenizer('models/tokenizer/tokenizer.json')
    print(tokenizer.decode(tokenizer.encode('what are you <mask>')))
    print('vocab size:', tokenizer.vocab_size)
    