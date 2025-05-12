import os
import json
from typing import Any, Iterable, Optional


class BPE_Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Optional[list[str]] = None):
        self.vocab = vocab
        self.merges = merges
        self.vocab_to_id = {v: k for k, v in self.vocab.items()}
        # Create a dictionary of mergeable pairs for faster lookup
        self.merge_dict = {(p1, p2): True for p1, p2 in self.merges}
        self.special_tokens = special_tokens

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs using the BPE tokenizer.

        Args:
            text: str
                The string to encode.

        Returns:
            list[int]
                The list of token IDs.
        """
        import regex as re
        ids = []

        # Special tokens processing
        if self.special_tokens:
            # Create a pattern that matches any of the special tokens
            pattern = "(" + '|'.join(sorted([re.escape(token) for token in self.special_tokens],key = lambda x: len(x),reverse=True)) + ")"
            # Split by special tokens keeping the delimiters
            parts = re.split(pattern, text)
        else:
            parts = [text]


        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Process each part
        for part in parts:

            if self.special_tokens and part in self.special_tokens:
                # Special tokens are added directly
                ids.append(self.vocab_to_id[part.encode("utf-8")])
            elif part:  # Skip empty parts
                # Tokenize regular text with BPE
                try:
                    tokens = re.findall(PAT, part)
                    for token in tokens:
                        id_token = self._bpe_encode(token)
                        ids.extend(id_token)
                except Exception as e:
                    print(f"Error encoding token: {token}")
                    print(f"Error message: {e}")
                    raise e

        return ids

    def _bpe_encode(self, text: str) -> list[int]:
        """
        Apply BPE encoding to a piece of text.
        """
        # Start with bytes representation
        byte_encoded = text.encode("utf-8")
        # Convert to list of integers (0-255)
        chars = [bytes([b]) for b in byte_encoded]
        
        # Apply merges in the order they were created
        while len(chars) > 1:
            # Find the highest-priority (earliest) merge
            pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
            merged = False
            
            # Check for mergeable pairs in the order they appear in self.merges
            for first, second in self.merges:
                # Look for this merge pair in the current sequence
                for i, pair in enumerate(pairs):
                    if pair[0] == first and pair[1] == second:
                        # Apply the merge
                        chars[i] = first + second
                        del chars[i+1]
                        merged = True
                        break
                if merged:
                    break
            
            # If no merge was applied, we're done
            if not merged:
                break
        
        # Convert to token IDs
        return [self.vocab_to_id[c] for c in chars]

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs into a string using the BPE tokenizer.

        Args:
            ids: list[int]  
                The list of token IDs.

        Returns:
            str
                The decoded string.
        """ 
        bytes_list = []
        for id in ids:
            bytes_list.append(self.vocab[id])
        try:
            # Try decoding as UTF-8
            return b''.join(bytes_list).decode('utf-8')
        except UnicodeDecodeError:
            # Fallback to replace invalid characters
            return b''.join(bytes_list).decode('utf-8', errors='replace')
    
    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens_path: str = None):
        """
        Initialize a BPE tokenizer from files.
        """
        # Load vocabulary
        with open(vocab_path, "r") as f:
            vocab_json = json.load(f)
        # 将字符串键转换回整数键，将字符串值转换回bytes
        vocab = {int(k): v.encode("latin1") for k, v in vocab_json.items()}
        
        # Load merges from JSON format
        merges = []
        with open(merges_path, "r") as f:
            serializable_merges = json.load(f)
            for first_bytes, second_bytes in serializable_merges:
                # 将整数列表转换回bytes
                first = bytes(first_bytes)
                second = bytes(second_bytes)
                merges.append((first, second))
        
        # Load special tokens if provided
        special_tokens = None
        if special_tokens_path:
            with open(special_tokens_path, "r") as f:
                special_tokens = [line.strip() for line in f if line.strip()]
        else:
            # Use the last tokens in vocab as special tokens (common convention)
            highest_id = max(vocab.keys())
            special_tokens = [vocab[highest_id].decode('latin1', errors='replace')]
        
        return cls(vocab, merges, special_tokens)
            
    def encode_iterable(self, texts: Iterable[str]) -> list[list[int]]:
        """
        Encode a list of strings into a list of lists of token IDs using the BPE tokenizer.
        """
        result = []

        for text in texts:
            result.extend(self.encode(text))
        return result
      
def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return BPE_Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # Initialize base vocabulary with byte values (0-255)
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = len(vocab)
    
    # Pre-tokenization
    import regex as re
    from collections import Counter
    
    # Load text data
    with open(input_path, "r", encoding="utf-8") as f:
        text_data = f.read()

    # Split on special tokens to ensure no merging across boundaries
    import re as std_re
    pattern = '|'.join(std_re.escape(token) for token in special_tokens)
    
    # If special tokens exist, use re.split to correctly split the data
    text_chunks = []
    if pattern:
        parts = std_re.split(f'({pattern})', text_data)
        # Keep only non-special token parts
        text_chunks = [part for part in parts if part not in special_tokens]
    else:
        text_chunks = [text_data]
    
    # Tokenize text using regex pattern for words, numbers, and other characters
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    # Dictionary to store token frequencies
    token_freqs = Counter()
    
    # Process each chunk independently
    for chunk in text_chunks:
        if not chunk:  # Skip empty chunks
            continue
        # Use finditer for memory efficiency
        for match in re.finditer(PAT, chunk):
            token = match.group(0)
            token_freqs[token] += 1
    
    # Convert tokens to byte sequences
    token_bytes = {}
    for token in token_freqs:
        token_bytes[token] = [bytes([b]) for b in token.encode("utf-8")]
    
    # BPE training loop
    merges = []
    
    # Calculate number of merge operations needed
    num_merges = vocab_size - len(vocab) - len(special_tokens)
    
    for _ in range(num_merges):
        # Count pair frequencies
        pair_freqs = Counter()
        
        for token, freq in token_freqs.items():
            bytes_seq = token_bytes[token]
            if len(bytes_seq) < 2:
                continue
                
            for i in range(len(bytes_seq) - 1):
                pair = (bytes_seq[i], bytes_seq[i+1])
                pair_freqs[pair] += freq
        
        if not pair_freqs:
            break
            
        # Find most frequent pair
        max_freq = max(pair_freqs.values())
        best_pairs = [pair for pair, freq in pair_freqs.items() if freq == max_freq]
        # Sort by lexicographical order to break ties
        best_pair = max(best_pairs)
        
        # Add merge to list
        first_token, second_token = best_pair
        merges.append((first_token, second_token))
        
        # Create new token by concatenating the bytes
        new_token = first_token + second_token
        vocab[next_token_id] = new_token
        
        # Update token byte sequences
        for token in token_bytes:
            i = 0
            new_bytes = []
            bytes_seq = token_bytes[token]
            
            while i < len(bytes_seq):
                if i < len(bytes_seq) - 1 and (bytes_seq[i], bytes_seq[i+1]) == best_pair:
                    new_bytes.append(new_token)
                    i += 2
                else:
                    new_bytes.append(bytes_seq[i])
                    i += 1
                    
            token_bytes[token] = new_bytes
            
        next_token_id += 1
    
    # Add special tokens to vocabulary
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1
    
    return vocab, merges