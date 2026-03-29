# src/bbpe.py
import re
from collections import defaultdict
import json
import time

def bytes_to_unicode():
    """Maps raw bytes (0-255) to visible unicode characters."""
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class BBPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = {} # <--- FIX 1: Added initialization here

    def get_word_frequencies(self, text):
        word_freqs = defaultdict(int)
        # Simple word split
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        for word in words:
            # 1. Convert word to raw bytes
            word_bytes = word.encode("utf-8")
            # 2. Map bytes to our safe unicode characters
            word_str = ''.join(self.byte_encoder[b] for b in word_bytes)
            # 3. Add boundaries
            chars = ' '.join(list(word_str)) + ' </w>'
            word_freqs[chars] += 1
        return word_freqs

    def get_pair_statistics(self, word_freqs):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, word_freqs):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in word_freqs:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = word_freqs[word]
        return v_out

    def train(self, corpus_text, target_vocab_size):
        print(f"Starting BBPE Training for target size: {target_vocab_size}...")
        start_time = time.time()
        word_freqs = self.get_word_frequencies(corpus_text)
        
        initial_tokens = set()
        for word in word_freqs.keys():
            for char in word.split():
                initial_tokens.add(char)
        
        current_vocab_size = len(initial_tokens)
        num_merges = target_vocab_size - current_vocab_size
        
        for i in range(num_merges):
            pairs = self.get_pair_statistics(word_freqs)
            if not pairs: break
            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = ''.join(best_pair)
            word_freqs = self.merge_vocab(best_pair, word_freqs)

        self.vocab = {token: idx for idx, token in enumerate(initial_tokens)}
        for pair, merged in self.merges.items():
            self.vocab[merged] = len(self.vocab)

        # <--- FIX 2: Generate ranks right after training completes --->
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges.keys())}
        
        return time.time() - start_time

    def save(self, filepath):
        data = {
            'vocab': self.vocab,
            'merges': {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.merges = {}
        self.bpe_ranks = {}
        for i, (key, value) in enumerate(data['merges'].items()):
            pair = tuple(key.split(' '))
            self.merges[pair] = value
            self.bpe_ranks[pair] = i 

    def encode(self, text):
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        encoded_tokens = []
        total_words = len(words)

        for word in words:
            # Apply byte encoding to the input text first!
            word_bytes = word.encode("utf-8")
            word_str = ''.join(self.byte_encoder[b] for b in word_bytes)
            
            word_symbols = list(word_str) + ['</w>']
            while len(word_symbols) > 1:
                pairs = [(word_symbols[i], word_symbols[i+1]) for i in range(len(word_symbols)-1)]
                pair_to_merge = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
                if pair_to_merge not in self.merges: break
                    
                new_symbols = []
                i = 0
                while i < len(word_symbols):
                    if i < len(word_symbols) - 1 and (word_symbols[i], word_symbols[i+1]) == pair_to_merge:
                        new_symbols.append(self.merges[pair_to_merge])
                        i += 2
                    else:
                        new_symbols.append(word_symbols[i])
                        i += 1
                word_symbols = new_symbols

            # Notice there is NO <UNK> check here. BBPE never generates OOV!
            for symbol in word_symbols:
                encoded_tokens.append(symbol)

        return {
            "tokens": encoded_tokens,
            "tokens_per_word": len(encoded_tokens) / total_words if total_words > 0 else 0,
            "oov_rate": 0.0 # BBPE by definition has 0 OOV rate
        }