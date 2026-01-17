"""
Custom Chess Tokenizer for the Chess Challenge.

This tokenizer treats each move as a single token using the extended UCI notation
from the Lichess dataset (e.g., WPe2e4, BNg8f6).

The dataset format uses:
- W/B prefix for White/Black
- Piece letter: P=Pawn, N=Knight, B=Bishop, R=Rook, Q=Queen, K=King
- Source and destination squares (e.g., e2e4)
- Special suffixes: (x)=capture, (+)=check, (+*)=checkmate, (o)/(O)=castling
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from transformers import PreTrainedTokenizer


class FrequencyChessTokenizer(PreTrainedTokenizer):
    """
    A frequency-based tokenizer for chess moves using extended UCI notation.
    
    This tokenizer maps each possible chess move to a unique token ID.
    The vocabulary is built from the training dataset to ensure all moves
    encountered during training have a corresponding token.
    
    Only includes moves that appear at least `min_frequency` times in the dataset.
    Rare moves become [UNK] tokens.
    
    Example:
        >>> tokenizer = FrequencyChessTokenizer()
        >>> tokenizer.encode("WPe2e4 BPe7e5")
        [1, 42, 87, 2]  # [BOS, e2e4, e7e5, EOS]
    """
    
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}
    
    # Special tokens
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        vocab: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        """
        Initialize the chess tokenizer.
        
        Args:
            vocab_file: Path to a JSON file containing the vocabulary mapping.
            vocab: Dictionary mapping tokens to IDs (alternative to vocab_file).
            **kwargs: Additional arguments passed to PreTrainedTokenizer.
        """
        # Initialize special tokens
        self._pad_token = self.PAD_TOKEN
        self._bos_token = self.BOS_TOKEN
        self._eos_token = self.EOS_TOKEN
        self._unk_token = self.UNK_TOKEN

        # Remove any duplicate special-token entries passed through kwargs
        # to avoid "multiple values for keyword" errors when loading from disk.
        kwargs.pop("pad_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("unk_token", None)
        
        # Load or create vocabulary
        if vocab is not None:
            self._vocab = vocab
        elif vocab_file is not None and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self._vocab = json.load(f)
        else:
            # Create a minimal vocabulary with just special tokens
            # The full vocabulary should be built from the dataset
            self._vocab = self._create_default_vocab()
        
        # Create reverse mapping
        self._ids_to_tokens = {v: k for k, v in self._vocab.items()}
        
        # Call parent init AFTER setting up vocab
        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs,
        )
    
    def _create_default_vocab(self) -> Dict[str, int]:
        """
        Create a minimal default vocabulary with just special tokens.
        
        For the full vocabulary, use `build_vocab_from_dataset()`.
        This minimal vocab is just a placeholder - you should build from data.
        """
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        vocab = {token: idx for idx, token in enumerate(special_tokens)}
        return vocab
    
    @classmethod
    def build_vocab_from_iterator(
        cls,
        iterator,
        min_frequency: int = 1,
    ) -> "FrequencyChessTokenizer":
        """
        Build a tokenizer vocabulary from an iterator of game strings.
        
        Args:
            iterator: An iterator yielding game strings (space-separated moves).
            min_frequency: Minimum frequency for a token to be included.
        
        Returns:
            A FrequencyChessTokenizer with the built vocabulary.
        """
        from collections import Counter
        
        token_counts = Counter()
        
        for game in iterator:
            moves = game.strip().split()
            token_counts.update(moves)
        
        # Filter by frequency
        tokens = [
            token for token, count in token_counts.items()
            if count >= min_frequency
        ]
        
        # Sort for reproducibility
        tokens = sorted(tokens)
        
        # Build vocabulary
        special_tokens = [cls.PAD_TOKEN, cls.BOS_TOKEN, cls.EOS_TOKEN, cls.UNK_TOKEN]
        vocab = {token: idx for idx, token in enumerate(special_tokens + tokens)}
        
        return cls(vocab=vocab)
    
    @classmethod
    def build_vocab_from_dataset(
        cls,
        dataset_name: str = "dlouapre/lichess_2025-01_1M",
        split: str = "train",
        column: str = "text",
        min_frequency: int = 500,
        max_samples: Optional[int] = 100000,
    ) -> "FrequencyChessTokenizer":
        """
        Build a tokenizer vocabulary from a Hugging Face dataset.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub.
            split: Dataset split to use.
            column: Column containing the game strings.
            min_frequency: Minimum frequency for a token to be included (default: 500).
            max_samples: Maximum number of samples to process (default: 100k).
        
        Returns:
            A FrequencyChessTokenizer with the built vocabulary.
        """
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        def game_iterator():
            for example in dataset:
                yield example[column]
        
        return cls.build_vocab_from_iterator(game_iterator(), min_frequency=min_frequency)
    
    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self._vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary as a dictionary."""
        return dict(self._vocab)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string of moves into a list of tokens.
        
        Args:
            text: A string of space-separated moves.
        
        Returns:
            List of move tokens.
        """
        return text.strip().split()
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its ID."""
        return self._vocab.get(token, self._vocab.get(self.UNK_TOKEN, 0))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token."""
        return self._ids_to_tokens.get(index, self.UNK_TOKEN)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a list of tokens back to a string."""
        # Filter out special tokens for cleaner output
        special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
        return " ".join(t for t in tokens if t not in special)
    
    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> tuple:
        """
        Save the vocabulary to a JSON file.
        
        Args:
            save_directory: Directory to save the vocabulary.
            filename_prefix: Optional prefix for the filename.
        
        Returns:
            Tuple containing the path to the saved vocabulary file.
        """
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )
        
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)


def count_vocab_from_dataset(
    dataset_name: str = "dlouapre/lichess_2025-01_1M",
    split: str = "train",
    column: str = "text",
    max_samples: Optional[int] = 10000,
) -> Dict[str, int]:
    """
    Count token frequencies in a dataset (useful for vocabulary analysis).
    
    Args:
        dataset_name: Name of the dataset on Hugging Face Hub.
        split: Dataset split to use.
        column: Column containing the game strings.
        max_samples: Maximum number of samples to process.
    
    Returns:
        Dictionary mapping tokens to their frequencies.
    """
    from collections import Counter
    from datasets import load_dataset
    
    dataset = load_dataset(dataset_name, split=split)
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    token_counts = Counter()
    
    for example in dataset:
        moves = example[column].strip().split()
        token_counts.update(moves)
    
    return dict(token_counts)


class ChessTokenizer(FrequencyChessTokenizer):
    """
    A compositional tokenizer for chess moves using colored pieces + positional markers.
    
    This tokenizer breaks each move into 5 core components with explicit structure:
    1. ColoredPiece: WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK
    2. SOURCE marker: [SOURCE]
    3. Source square: a1, a2, ..., h8
    4. DEST marker: [DEST]
    5. Destination square: a1, a2, ..., h8
    
    Optional modifier tokens for captures, checks, checkmate, and castling.
    
    Example:
        >>> tokenizer = ChessTokenizer()
        >>> tokenizer.encode("WPe2e4 BPe7e5")
        [1, WP_id, SRC_id, e2_id, DST_id, e4_id, BP_id, SRC_id, e7_id, DST_id, e5_id, 2]
    
    Vocabulary:
    - Colored pieces (12): WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK
    - Position markers (2): [SOURCE], [DEST]
    - Squares (64): a1-h8
    - Modifiers (5): [CAPTURE], [CHECK], [CHECKMATE], [CASTLING_KS], [CASTLING_QS]
    - Special (4): [PAD], [BOS], [EOS], [UNK]
    Total: ~87 tokens (deterministic, no frequency filtering)
    """
    
    # Colored piece tokens (color + piece)
    COLORED_PIECES = [
        'WP', 'WN', 'WB', 'WR', 'WQ', 'WK',  # White pieces
        'BP', 'BN', 'BB', 'BR', 'BQ', 'BK',  # Black pieces
    ]
    
    # Position markers
    POSITION_MARKERS = ['[SOURCE]', '[DEST]']
    
    # Board squares (standard chess notation)
    SQUARES = [f"{file}{rank}" for rank in range(1, 9) for file in "abcdefgh"]
    
    # Move modifiers
    MODIFIERS = ['[CAPTURE]', '[CHECK]', '[CHECKMATE]', '[CASTLING_KS]', '[CASTLING_QS]']
    
    def __init__(self, **kwargs):
        """
        Initialize the compositional chess tokenizer.
        
        Vocabulary is built deterministically from pieces and squares.
        No vocab_file or dataset scanning needed.
        """
        # Remove vocab-related kwargs to avoid conflicts
        kwargs.pop("vocab_file", None)
        kwargs.pop("vocab", None)
        
        # Build deterministic vocabulary
        vocab = self._build_deterministic_vocab()
        
        # Initialize parent with the built vocab
        super().__init__(vocab=vocab, **kwargs)
    
    def _build_deterministic_vocab(self) -> Dict[str, int]:
        """
        Build vocabulary deterministically from colored pieces, squares, and modifiers.
        
        Returns:
            Dictionary mapping token strings to IDs.
        """
        vocab = {}
        idx = 0
        
        # Special tokens first (matching parent class order)
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for token in special_tokens:
            vocab[token] = idx
            idx += 1
        
        # Colored piece tokens
        for piece in self.COLORED_PIECES:
            vocab[piece] = idx
            idx += 1
        
        # Position marker tokens
        for marker in self.POSITION_MARKERS:
            vocab[marker] = idx
            idx += 1
        
        # Square tokens
        for square in self.SQUARES:
            vocab[square] = idx
            idx += 1
        
        # Modifier tokens
        for modifier in self.MODIFIERS:
            vocab[modifier] = idx
            idx += 1
        
        return vocab
    
    def _parse_move(self, move_str: str) -> Dict:
        """
        Parse a move string in extended UCI notation.
        
        Args:
            move_str: Move string like "WPe2e4" or "BNg8f6(x)" or "We1g1(o)"
        
        Returns:
            Dictionary with keys: piece, color, src, dest, modifiers
        """
        import re
        
        # Pattern: [WB][PNBRQK]<square><square>(<modifiers>)
        pattern = r'([WB])([PNBRQK])([a-h][1-8])([a-h][1-8])((?:\([^)]*\))?)'
        match = re.match(pattern, move_str.strip())
        
        if not match:
            raise ValueError(f"Invalid move format: {move_str}")
        
        color, piece, src, dest, modifier_str = match.groups()
        
        # Parse modifiers
        modifiers = []
        if modifier_str:
            # Remove parentheses and split by lowercase letters/symbols
            mod_content = modifier_str.strip('()')
            
            if 'x' in mod_content:
                modifiers.append('[CAPTURE]')
            if '+*' in mod_content:
                modifiers.append('[CHECKMATE]')
            elif '+' in mod_content:
                modifiers.append('[CHECK]')
            if 'o' in mod_content or 'O' in mod_content:
                # Determine kingside vs queenside based on destination
                if dest == 'g1' or dest == 'g8':
                    modifiers.append('[CASTLING_KS]')
                elif dest == 'c1' or dest == 'c8':
                    modifiers.append('[CASTLING_QS]')
        
        return {
            'piece': piece,
            'color': color,
            'src': src,
            'dest': dest,
            'modifiers': modifiers,
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string of moves into component tokens with positional markers.
        
        Each move becomes: [ColoredPiece, [SOURCE], source, [DEST], dest, *modifiers]
        
        Args:
            text: String of space-separated moves (e.g., "WPe2e4 BPe7e5")
        
        Returns:
            List of component tokens with structure markers.
        """
        move_strings = text.strip().split()
        tokens = []
        
        for move_str in move_strings:
            parsed = self._parse_move(move_str)
            
            # Build colored piece token (color + piece)
            colored_piece = f"{parsed['color']}{parsed['piece']}"
            tokens.append(colored_piece)
            
            # Add positional markers and squares
            tokens.append('[SOURCE]')
            tokens.append(parsed['src'])
            tokens.append('[DEST]')
            tokens.append(parsed['dest'])
            
            # Add modifier tokens if any
            tokens.extend(parsed['modifiers'])
        
        return tokens
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """
        Reconstruct moves from component tokens with positional markers.
        
        Expects structure: ColoredPiece, [SOURCE], source, [DEST], dest, *modifiers
        
        Args:
            tokens: List of component tokens
        
        Returns:
            Space-separated move string.
        """
        moves = []
        token_idx = 0
        
        while token_idx < len(tokens):
            token = tokens[token_idx]
            
            # Skip special tokens
            special = {self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN}
            if token in special:
                token_idx += 1
                continue
            
            # Expect: ColoredPiece token
            if token not in self.COLORED_PIECES:
                break
            
            colored_piece = token
            color = colored_piece[0]  # W or B
            piece = colored_piece[1]  # P, N, B, R, Q, K
            
            # Expect: [SOURCE] marker
            if token_idx + 1 >= len(tokens) or tokens[token_idx + 1] != '[SOURCE]':
                break
            
            # Expect: source square
            if token_idx + 2 >= len(tokens):
                break
            src = tokens[token_idx + 2]
            if src not in self.SQUARES:
                break
            
            # Expect: [DEST] marker
            if token_idx + 3 >= len(tokens) or tokens[token_idx + 3] != '[DEST]':
                break
            
            # Expect: dest square
            if token_idx + 4 >= len(tokens):
                break
            dest = tokens[token_idx + 4]
            if dest not in self.SQUARES:
                break
            
            # Build move string
            move_str = f"{color}{piece}{src}{dest}"
            
            # Collect modifiers (next tokens until we hit another colored piece or end)
            token_idx += 5
            modifiers_list = []
            
            while token_idx < len(tokens) and tokens[token_idx] in self.MODIFIERS:
                modifier = tokens[token_idx]
                modifiers_list.append(modifier)
                token_idx += 1
            
            # Append modifier suffixes
            if modifiers_list:
                modifier_str = ""
                if '[CAPTURE]' in modifiers_list:
                    modifier_str += "x"
                if '[CHECKMATE]' in modifiers_list:
                    modifier_str += "+*"
                elif '[CHECK]' in modifiers_list:
                    modifier_str += "+"
                if '[CASTLING_KS]' in modifiers_list:
                    modifier_str += "o"
                elif '[CASTLING_QS]' in modifiers_list:
                    modifier_str += "o"
                
                move_str += f"({modifier_str})"
            
            moves.append(move_str)
        
        return " ".join(moves)


class ChessLogitsProcessor:
    """
    Logits processor for enforcing chess move structure during generation.
    
    Enforces the token sequence pattern:
    ColoredPiece [SOURCE] source [DEST] dest [modifiers]*
    
    Uses a state machine with 6 states:
    - State 0: Expect colored piece (WP, WN, ..., BK)
    - State 1: Expect [SOURCE] marker
    - State 2: Expect source square (a1-h8)
    - State 3: Expect [DEST] marker
    - State 4: Expect dest square (a1-h8)
    - State 5: Expect modifiers or next colored piece
    
    Token structure is hardcoded to match ChessTokenizer:
    - Colored pieces: WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK
    - Position markers: [SOURCE], [DEST]
    - Squares: a1-h8 (64 total)
    - Modifiers: [CAPTURE], [CHECK], [CHECKMATE], [CASTLING_KS], [CASTLING_QS]
    """
    
    # Token vocabulary indices (hardcoded to match ChessTokenizer vocab order)
    # Special tokens: [PAD]=0, [BOS]=1, [EOS]=2, [UNK]=3
    # Colored pieces (4-15)
    COLORED_PIECE_IDS = {
        'WP': 4, 'WN': 5, 'WB': 6, 'WR': 7, 'WQ': 8, 'WK': 9,
        'BP': 10, 'BN': 11, 'BB': 12, 'BR': 13, 'BQ': 14, 'BK': 15
    }
    # Position markers (16-17)
    POSITION_MARKER_IDS = {'[SOURCE]': 16, '[DEST]': 17}
    # Squares (18-81): a1=18, a2=19, ..., h8=81
    SQUARE_IDS = {f"{file}{rank}": 18 + (rank - 1) * 8 + ord(file) - ord('a')
                  for rank in range(1, 9) for file in "abcdefgh"}
    # Modifiers (82-86)
    MODIFIER_IDS = {
        '[CAPTURE]': 82, '[CHECK]': 83, '[CHECKMATE]': 84,
        '[CASTLING_KS]': 85, '[CASTLING_QS]': 86
    }
    
    def __init__(self):
        """
        Initialize the logits processor with hardcoded ChessTokenizer structure.
        """
        import torch
        self.torch = torch
        
        # Convert to sets for membership testing
        self.colored_piece_ids = set(self.COLORED_PIECE_IDS.values())
        self.square_ids = set(self.SQUARE_IDS.values())
        self.modifier_ids = set(self.MODIFIER_IDS.values())
    
    def _get_state(self, input_ids):
        """
        Determine current state in move sequence based on recent tokens.
        
        Returns state (0-5) indicating what token type is expected next.
        """
        if input_ids.numel() == 0:
            return 0  # Start: expect colored piece
        
        # Get the last token
        last_token_id = input_ids[0, -1].item()
        
        # Count back to find the last colored piece
        seq = input_ids[0].tolist()
        
        # Find pattern of most recent tokens
        # Work backwards to identify state
        for i in range(len(seq) - 1, -1, -1):
            token_id = seq[i]
            
            # If we see a colored piece, analyze what comes after
            if token_id in self.colored_piece_ids:
                # Count tokens after this piece
                tokens_after = len(seq) - 1 - i
                
                # Pattern: piece, [SOURCE], source, [DEST], dest, ...
                if tokens_after == 0:
                    return 1  # Expect [SOURCE]
                elif tokens_after == 1:
                    # Check if last token is [SOURCE]
                    if seq[-1] == self.POSITION_MARKER_IDS['[SOURCE]']:
                        return 2  # Expect source square
                    else:
                        return 1  # Unexpected, reset to [SOURCE]
                elif tokens_after == 2:
                    # Should be: piece, [SOURCE], source
                    if seq[-2] == self.POSITION_MARKER_IDS['[SOURCE]'] and seq[-1] in self.square_ids:
                        return 3  # Expect [DEST]
                    else:
                        return 1  # Reset
                elif tokens_after == 3:
                    # Should be: piece, [SOURCE], source, [DEST]
                    if seq[-1] == self.POSITION_MARKER_IDS['[DEST]']:
                        return 4  # Expect dest square
                    else:
                        return 1  # Reset
                elif tokens_after == 4:
                    # Should be: piece, [SOURCE], source, [DEST], dest
                    if (seq[-3] == self.POSITION_MARKER_IDS['[DEST]'] and 
                        seq[-2] in self.square_ids and
                        seq[-1] in self.square_ids):
                        return 5  # Expect modifiers or next piece
                    else:
                        return 1  # Reset
                else:
                    # tokens_after >= 5: in modifier section
                    # Check if we're building modifiers
                    return 5
        
        # No colored piece found, start fresh
        return 0
    
    def constrain_logits(self, input_ids, logits):
        """
        Mask invalid tokens based on current state in move sequence.
        
        Args:
            input_ids: Current token sequence (batch_size, seq_len)
            logits: Logits from model (batch_size, vocab_size)
        
        Returns:
            logits with invalid tokens masked to -inf
        """
        logits = logits.clone()  # Don't modify in place
        state = self._get_state(input_ids)
        
        # Create mask: -inf for invalid tokens, 0 for valid
        mask = self.torch.full_like(logits, float('-inf'))
        
        if state == 0 or state == 5:
            # State 0 (start) or State 5 (after move): expect colored piece
            mask[:, list(self.colored_piece_ids)] = 0
        elif state == 1:
            # Expect [SOURCE] marker
            mask[:, self.POSITION_MARKER_IDS['[SOURCE]']] = 0
        elif state == 2:
            # Expect source square
            mask[:, list(self.square_ids)] = 0
        elif state == 3:
            # Expect [DEST] marker
            mask[:, self.POSITION_MARKER_IDS['[DEST]']] = 0
        elif state == 4:
            # Expect dest square
            mask[:, list(self.square_ids)] = 0
        
        # Apply mask: keep valid tokens, zero out invalid ones
        logits[mask == float('-inf')] = float('-inf')
        
        return logits

