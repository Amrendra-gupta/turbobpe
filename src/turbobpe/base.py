"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
import unicodedata

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts
    
def encode_get_stats(ids):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids
    

def find_overlapping_cases(coordinates):
    """
    Given a list of (A, B) pairs, return all pairs that form a chain overlap.
    A chain overlap occurs when the right token of one pair is the left token of another: (A, B) and (B, C).
    Merging both in the same batch would be unsafe as the first merge affects the second.
    Example: [(1, 2), (2, 3), (4, 5)] -> [(1, 2), (2, 3)]  # (4, 5) is safe, no chain
    """
    overlapping_cases = []
    coordinate_dict = {}
    
    for coord in coordinates:
        if coord[0] not in coordinate_dict:
            coordinate_dict[coord[0]] = []
        coordinate_dict[coord[0]].append(coord[1])
    
    # Check for (A, B) and (B, C) pairs
    for a, b_list in coordinate_dict.items():
        for b in b_list:
            if b in coordinate_dict:
                for c in coordinate_dict[b]:
                    overlapping_cases.append((a, b))
                    overlapping_cases.append((b, c))
                    
    return overlapping_cases

def filter_top_pairs(top_pairs, overlaps):
    """
    Given a sorted (descending by frequency) list of top pairs and their overlapping cases,
    return a safe subset for batch merging by dropping chained overlaps.
    The first pair in an overlap chain is kept (highest count, safe to merge), the second is dropped.
    Example: top_pairs=[(1,2),(2,3),(4,5)], overlaps=[(1,2),(2,3)] -> [(1,2),(4,5)]
    """
    common_count = 0
    result = []
    for pair in top_pairs:
        # Check if the pair is in overlaps
        if pair in overlaps:
            common_count += 1
        # Stop when the second common element is found
        if common_count == 2:
            break
        result.append(pair)   
    return result

def batch_merge(ids, pairs, indices):
    """
    Sequentially applies a batch of non-overlapping pair merges 
    to a single ID sequence.
    """
    newids = list(ids)
    n = len(pairs)
    for i in range(n):
        p0, p1 = pairs[i]
        idx = indices[i]
        j = 0
        limit = len(newids) - 1  # cache once per pair
        while j < limit:
            if newids[j] == p0 and newids[j + 1] == p1:
                newids[j] = idx
                del newids[j + 1]
                limit -= 1  # list shrank, update limit
                j += 1
            else:
                j += 1
        if limit == 0:   # list is now length 1
            return None
    return newids


def training(num_merges, ids, batch_size, verbose):
    # iteratively merge the most common pairs to create new tokens
    merges = {} # (int, int) -> int
    vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
    next_id = 256
    while next_id - 256 < num_merges:
        # count the number of times every consecutive pair appears
        stats = {}
        for chunk_ids in ids:
            # passing in stats will update it in place, adding up counts
            get_stats(chunk_ids, stats)
        # pairs we attempt to merge at once (e.g., 10) with highest occurance
        top_pairs = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:batch_size]
        top_pairs = [pair[0] for pair in top_pairs]
        # Filter out structural chain overlaps
        overlaps = find_overlapping_cases(top_pairs)
        if len(overlaps)>0:
            top_pairs = filter_top_pairs(top_pairs, overlaps)
            
        # Truncate top_pairs if it would overshoot the exact num_merges limit
        remaining_slots = num_merges - (next_id - 256)
        if len(top_pairs) > remaining_slots:
            top_pairs = top_pairs[:remaining_slots]
                
        # mint a new token: assign it the next available id
        idx = [next_id + j for j in range(len(top_pairs))]
        next_id += len(idx) 
        # replace all occurrences of pair in0 ids with idx
        ids = [res for chunk_ids in ids if (res := batch_merge(chunk_ids, top_pairs, idx)) is not None]   
        
        # save the merge and print details
        for pair, new_idx in zip(top_pairs, idx):
            merges[pair] = new_idx
            vocab[new_idx] = vocab[pair[0]] + vocab[pair[1]]
            
            if verbose:
                print(f"merge {new_idx-256}: {pair} -> {new_idx} ({vocab[new_idx]}) had {stats[pair]} occurrences")
    return merges, vocab


def _encode_chunk(merges, text_bytes):
    # return the token ids
    # let's begin. first, convert all bytes to integers in range 0..255
    ids = list(text_bytes)
    while len(ids) >= 2:
        # find the pair with the lowest merge index
        stats = encode_get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        # subtle: if there are no more merges available, the key will
        # result in an inf for every single pair, and the min will be
        # just the first pair in the list, arbitrarily
        # we can detect this terminating case by a membership check
        if pair not in merges:
            break # nothing else can be merged anymore
        # otherwise let's merge the best pair (lowest merge index)
        idx = merges[pair]
        ids = merge(ids, pair, idx)
    return ids


# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()