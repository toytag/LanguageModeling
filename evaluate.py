import torch
import torch.nn.functional as F

from tokenizer import WordPieceTokenizer
from model import LanguageModel


if __name__ == '__main__':
    
    tokenizer = WordPieceTokenizer('models/tokenizer/tokenizer.json')

    checkpoint = torch.load('models/lm/latest.pth')
        
    model = LanguageModel(n_vocab=10000)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    input_tokens = tokenizer.encode('a king is a man and a queen is a <mask>')
    input_tokens = torch.LongTensor([input_tokens])
    input_mask = torch.ones_like(input_tokens)
    input_mask[0, -1] = 0

    with torch.no_grad():
        output, *_ = model(input_tokens, src_mask=input_mask)

    output = torch.argmax(F.softmax(output, dim=-1), dim=-1).squeeze(0).numpy()
    print(tokenizer.decode(output))