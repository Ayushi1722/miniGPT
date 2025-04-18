import os
import json
import requests
import regex as re

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))

def get_pairs(word):
    return set(zip(word, word[1:]))

class Encoder:
    def __init__(self, encoder, bpe_merges):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.encoder = encoder
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.pat = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")

    def bpe(self, token):
        if token in self.cache: return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs: return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks: break
            first, second = bigram
            new_word, i = [], 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1: break
            pairs = get_pairs(word)

        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_idx = []
        for token in re.findall(self.pat, text):
            token_bytes = token.encode('utf-8')
            token_translated = ''.join(self.byte_encoder[b] for b in token_bytes)
            token_merged = self.bpe(token_translated).split(' ')
            token_ix = [self.encoder[bpe_token] for bpe_token in token_merged]
            bpe_idx.extend(token_ix)
        return bpe_idx

    def decode(self, bpe_idx):
        tokens_merged = [self.decoder[token] for token in bpe_idx]
        tokens_flat = ''.join(tokens_merged)
        tokens_bytes = bytearray([self.byte_decoder[c] for c in tokens_flat])
        return tokens_bytes.decode('utf-8', errors='replace')

def get_encoder():
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'mingpt')
    os.makedirs(cache_dir, exist_ok=True)

    def get_file(local, remote):
        if not os.path.isfile(local):
            with open(local, 'wb') as f:
                f.write(requests.get(remote).content)

    enc_path = os.path.join(cache_dir, 'encoder.json')
    get_file(enc_path, 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json')
    with open(enc_path, 'r') as f: encoder = json.load(f)

    vocab_path = os.path.join(cache_dir, 'vocab.bpe')
    get_file(vocab_path, 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        bpe_merges = [tuple(merge_str.split()) for merge_str in f.read().split('\n')[1:-1]]

    return Encoder(encoder, bpe_merges)
