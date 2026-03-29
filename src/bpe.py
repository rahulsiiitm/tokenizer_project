# src/bpe.py
import re
from collections import defaultdict, Counter
import json
import time

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
        self.base_vocab_size = 0
        self.bpe_ranks = {} # <--- FIX 1: Initialized here

    def get_word_frequencies(self, text):
        """Pre-tokenizes text into words and adds the </w> boundary marker."""
        word_freqs = defaultdict(int)
        # Simple whitespace and punctuation split (Assumption: English corpus)
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        for word in words:
            # Represent word as space-separated characters with an end-of-word token
            chars = ' '.join(list(word)) + ' </w>'
            word_freqs[chars] += 1
        return word_freqs

    def get_pair_statistics(self, word_freqs):
        """Counts the frequency of all adjacent symbol pairs."""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, word_freqs):
        """Replaces the most frequent pair with the new merged symbol in all words."""
        v_out = {}
        bigram = re.escape(' '.join(pair))
        # Regex to match the exact pair with boundaries
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in word_freqs:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = word_freqs[word]
        return v_out

    def train(self, corpus_text, target_vocab_size):
        """Trains the BPE model up to the target vocabulary size."""
        print(f"Starting BPE Training for target size: {target_vocab_size}...")
        start_time = time.time()
        
        word_freqs = self.get_word_frequencies(corpus_text)
        
        # Initial vocabulary is just the unique characters + </w>
        initial_tokens = set()
        for word in word_freqs.keys():
            for char in word.split():
                initial_tokens.add(char)
        
        self.base_vocab_size = len(initial_tokens)
        current_vocab_size = self.base_vocab_size
        num_merges = target_vocab_size - current_vocab_size
        
        print(f"Base vocabulary size: {self.base_vocab_size}")
        print(f"Performing {num_merges} merges...")

        for i in range(num_merges):
            pairs = self.get_pair_statistics(word_freqs)
            if not pairs:
                break # No more pairs to merge
                
            # Get the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Record the merge
            self.merges[best_pair] = ''.join(best_pair)
            
            # Update the word dictionary
            word_freqs = self.merge_vocab(best_pair, word_freqs)
            
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1} merges...")

        # Build final vocabulary mapping
        self.vocab = {token: idx for idx, token in enumerate(initial_tokens)}
        for pair, merged in self.merges.items():
            self.vocab[merged] = len(self.vocab)

        # <--- FIX 2: Generate ranks directly after training so encode() works --->
        self.bpe_ranks = {pair: i for i, pair in enumerate(self.merges.keys())}

        exec_time = time.time() - start_time
        print(f"Training complete in {exec_time:.2f} seconds. Final Vocab Size: {len(self.vocab)}")
        return exec_time

    def save(self, filepath):
        """Saves the vocabulary and merges to a JSON file."""
        data = {
            'vocab': self.vocab,
            'merges': {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        """Loads a saved vocabulary and merges."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        
        # Reconstruct merges and assign a "rank" (priority) based on order
        self.merges = {}
        self.bpe_ranks = {}
        for i, (key, value) in enumerate(data['merges'].items()):
            pair = tuple(key.split(' '))
            self.merges[pair] = value
            self.bpe_ranks[pair] = i # Lower index = higher priority

    def encode(self, text):
        """Tokenizes text using the learned BPE merges and calculates metrics."""
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        encoded_tokens = []
        oov_count = 0
        total_words = len(words)

        for word in words:
            word_symbols = list(word) + ['</w>']
            
            while len(word_symbols) > 1:
                # Get all adjacent pairs
                pairs = [(word_symbols[i], word_symbols[i+1]) for i in range(len(word_symbols)-1)]
                
                # Find the pair that we learned earliest
                pair_to_merge = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
                
                # If the pair is not in our learned merges, we stop merging
                if pair_to_merge not in self.merges:
                    break
                    
                # Apply the merge
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

            # Check for OOV (Out of Vocabulary)
            for symbol in word_symbols:
                if symbol in self.vocab:
                    encoded_tokens.append(symbol)
                else:
                    encoded_tokens.append("<UNK>")
                    oov_count += 1

        return {
            "tokens": encoded_tokens,
            "tokens_per_word": len(encoded_tokens) / total_words if total_words > 0 else 0,
            "oov_rate": oov_count / len(encoded_tokens) if encoded_tokens else 0
        }