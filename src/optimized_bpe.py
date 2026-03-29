# src/optimized_bpe.py
import re
from bpe import BPETokenizer

class EdgeBPETokenizer(BPETokenizer):
    def __init__(self):
        super().__init__()
        self.fast_lookup = {}

    def prepare_for_edge(self):
        """Pre-computes merges into a fast hash map for O(1) lookups."""
        # Map pairs to their merged strings for fast access during encoding
        self.fast_lookup = {pair: merged for pair, merged in self.merges.items()}

    def encode(self, text):
        """Optimized encoding for edge devices avoiding slow global regex scans."""
        # Simple word split for efficiency
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        encoded_tokens = []
        
        for word in words:
            word_symbols = list(word) + ['</w>']
            while len(word_symbols) > 1:
                # Find the pair with the best (lowest) rank in one pass
                best_pair = None
                best_rank = float('inf')
                best_idx = -1
                
                for i in range(len(word_symbols) - 1):
                    pair = (word_symbols[i], word_symbols[i+1])
                    rank = self.bpe_ranks.get(pair, float('inf'))
                    if rank < best_rank:
                        best_rank, best_pair, best_idx = rank, pair, i
                
                if best_idx == -1: 
                    break # No more learned merges possible
                
                # Apply the merge instantly using the pre-computed lookup
                merged_val = self.fast_lookup[best_pair]
                word_symbols = word_symbols[:best_idx] + [merged_val] + word_symbols[best_idx+2:]
            
            encoded_tokens.extend(word_symbols)
            
        return {"tokens": encoded_tokens}