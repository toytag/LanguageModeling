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

        pattern = r'[a-zA-Z0-9\']+|[,<.>/?;:\'\"\[\{\]\}\\\|\-_=\+`~!@#$%^&*\(\)]'

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
        self._token2id_updated = False
        self.id2token = {}
        self._id2token_updated = False

        if vocab_file:
            self._update_token2id()
            self._update_id2token()


    def build_vocab(self, corpus):
        if self.lowercase:
            corpus = corpus.lower()
        
        tokens = self.p.findall(corpus)
        self.vocab += Counter(tokens)

        self._token2id_updated = False
        self._id2token_updated = False


    def _update_token2id(self):
        # 0 reserved, 1 for unknown token
        self.token2id = {special_token: i + 2 for i, special_token in enumerate(self.special_tokens)}
        
        for i, (vocab, freq) in enumerate(self.vocab.most_common()):
            i = i + len(self.special_tokens) + 2
            if freq >= self.min_freq and i < self.vocab_size_limit:
                self.token2id[vocab] = i

        self.vocab_size = len(self.token2id.keys()) + 2
        self._token2id_updated = True


    def _update_id2token(self):
        # 0 reserved, 1 for unknown token
        self.id2token = {i + 2: special_token for i, special_token in enumerate(self.special_tokens)}

        for i, (vocab, freq) in enumerate(self.vocab.most_common()):
            i = i + len(self.special_tokens) + 2
            if freq >= self.min_freq and i < self.vocab_size_limit:
                self.id2token[i] = vocab
                
        self.vocab_size = len(self.id2token.keys()) + 2
        self._id2token_updated = True


    def encode(self, text):
        if not self._token2id_updated:
            self._update_token2id()
        if self.lowercase:
            text = text.lower()
        return [self.token2id.get(token, 1) for token in self.p.findall(text)]


    def decode(self, ids):
        if not self._id2token_updated:
            self._update_id2token()
        return ' '.join([self.id2token.get(id_, '[UNK]') for id_ in ids])

    
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
    
    print(tokenizer.decode(tokenizer.encode('what are you <mask> fadskjfaf')))
    print('vocab size:', tokenizer.vocab_size)
    tokenizer.save('models/tokenizer/tokenizer.json')
    tokenizer = WordPieceTokenizer('models/tokenizer/tokenizer.json')
    print(tokenizer.decode(tokenizer.encode('what are you <mask> fadskjfaf')))
    print('vocab size:', tokenizer.vocab_size)

    print(text[:51], tokenizer.decode(tokenizer.encode(text[:51])))
    