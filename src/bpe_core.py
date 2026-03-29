import collections
import re
import json

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.merges = {}  # (char_a, char_b) -> merged_id
        self.vocab = {}   # id -> string representation (for decoding)
        
    def _get_stats(self, ids_list):
        counts = collections.defaultdict(int)
        for word_id_list in ids_list:
            for i in range(len(word_id_list) - 1):
                counts[(word_id_list[i], word_id_list[i+1])] += 1
        return counts

    def _merge(self, ids_list, pair, idx):
        new_ids_list = []
        for word_id_list in ids_list:
            new_word = []
            i = 0
            while i < len(word_id_list):
                if i < len(word_id_list) - 1 and (word_id_list[i], word_id_list[i+1]) == pair:
                    new_word.append(idx)
                    i += 2
                else:
                    new_word.append(word_id_list[i])
                    i += 1
            new_ids_list.append(new_word)
        return new_ids_list

    def train(self, text):
        words = text.split()
        # Initialize words as lists of character bytes + a special end-of-word space byte
        ids_list = [[ord(c) for c in word] + [ord(' ')] for word in words]
        
        # Populate initial vocabulary (bytes)
        for word in ids_list:
            for byte in word:
                self.vocab[byte] = chr(byte)
                
        current_vocab_size = len(self.vocab)
        num_merges = self.vocab_size - current_vocab_size
        
        for i in range(num_merges):
            stats = self._get_stats(ids_list)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            new_token_id = 256 + i
            
            self.merges[best_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            ids_list = self._merge(ids_list, best_pair, new_token_id)

    def encode(self, text):
        """Tokenize new text using learned merges."""
        words = text.split()
        ids_list = [[ord(c) for c in word] + [ord(' ')] for word in words]
        
        # Apply merges in the exact order they were learned
        # (This is a simplified greedy approach for standard BPE)
        for pair, new_id in self.merges.items():
            ids_list = self._merge(ids_list, pair, new_id)
            
        # Flatten the list of lists into a single token stream
        return [token for word_ids in ids_list for token in word_ids]

    def decode(self, ids):
        """Convert token IDs back to a string."""
        text = "".join([self.vocab.get(idx, "<UNK>") for idx in ids])
        return text.replace(" ", " ").strip() # Clean up the word boundary markers