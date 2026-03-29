# src/noise.py
import random
import re

class NoiseInjector:
    def __init__(self, seed=42):
        random.seed(seed)
        # Simulating common OCR mistakes from your previous assignment!
        self.ocr_map = {
            'l': 'I', 'I': 'l', 
            'O': '0', '0': 'O', 
            'rn': 'm', 'm': 'rn',
            'cl': 'd', 'd': 'cl',
            'vv': 'w', 'w': 'vv',
            'S': '5', '5': 'S'
        }

    def inject_spelling_typos(self, text, error_rate=0.1):
        """Swaps adjacent characters to simulate fast typing/spelling errors."""
        chars = list(text)
        num_errors = int(len(chars) * error_rate)
        
        for _ in range(num_errors):
            idx = random.randint(0, len(chars) - 2)
            # Swap character with the next one
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
            
        return "".join(chars)

    def inject_ocr_distortion(self, text, error_rate=0.15):
        """Replaces characters with visually similar ones."""
        words = text.split()
        corrupted_words = []
        
        for word in words:
            if random.random() < error_rate:
                for target, replacement in self.ocr_map.items():
                    if target in word:
                        word = word.replace(target, replacement, 1)
                        break 
            corrupted_words.append(word)
            
        return " ".join(corrupted_words)