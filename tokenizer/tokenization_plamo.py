import json
import math
import os
from shutil import copyfile
from typing import Any, Optional, Tuple

import numpy as np

# NOTE: numba does not support type hints for njit: https://github.com/python/mypy/issues/16149
from numba import njit  # type: ignore[attr-defined]
from numba.core import types
from numba.typed import Dict, List
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.jsonl"}
logger = logging.get_logger(__name__)

INVALID_SCORE = -20000000
UNKNOWN_SCORE = -10000000

TABLE_PIECE_LENGTH = 0
TABLE_TOKEN_ID = 1
TABLE_SCORE = 2
TABLE_PIECE_ID = 3

PATH_TOKEN_LENGTH = 0
PATH_TOKEN_ID = 1
PATH_NUM_TOKENS = 2


class AhoCorasick:
    def __init__(self) -> None:
        # List of tokens in the vocabulary.
        self._tokens: list[str]

        # A mapping from a byte code point to a token ID, used for byte fallback.
        self._bytes: np.ndarray

        # A mapping from a suffix's piece code to a suffix ID.
        #
        # Typically, the Aho-Corasick algorithm builds a Trie and adds suffix links between nodes
        # of the Trie. In this implementation, a suffix ID corresponds to a node in the trie, and
        # a piece code to an edge (in other words, a pair of a node and the next character).
        #
        # A piece code is a 64-bit integer:
        # - The upper 32 bits store the Unicode code point of the first character.
        # - The lower 32 bits store the suffix ID of the remaining suffix.
        #
        # A suffix ID is an integer indicating the starting position in the _table.
        self._to_suffix_id: Dict[types.int64, types.int32]

        # Flattened table representing the Trie structure for the Aho-Corasick algorithm.
        # It stores information including scores for each piece (prefix) within each suffix.
        # It is flattened for memory efficiency and performance. Suffixes are stored in
        # lexicographical order of their reversed strings, which improves memory access locality
        # when exploring new characters starting from the string's end. Pieces within a suffix are
        # stored in the decreasing order of their lengths.
        #
        # Each piece (a prefix fo the suffix) contains four pieces of information:
        # - TABLE_PIECE_LENGTH: Length of the piece.
        # - TABLE_TOKEN_ID: Token ID (or -1 if the piece is not a valid token).
        # - TABLE_SCORE: Score (or INVALID_SCORE if the piece is not a valid token).
        # - TABLE_PIECE_ID: Piece ID of the suffix.
        #
        # Each suffix also includes a sentinel row with a length of 1, a score of UNKNOWN_SCORE,
        # and a token ID of -1. Sentinel rows are identified by the score being UNKNOWN_SCORE.
        self._table: np.ndarray

    def build(self, vocab: list[Any]) -> None:
        self._bytes = np.zeros(256, dtype=np.int32)
        self._to_suffix_id = Dict.empty(key_type=types.int64, value_type=types.int32)

        # Build suffix_to_score and token_to_token_id.
        # The suffix_to_score dictionary maps a suffix to its score. It also includes all suffixes
        # of the token for the Trie structure for the Aho-Corasick algorithm. If a suffix is not a
        # valid token, its score is set to math.nan.
        # The token_to_token_id dictionary maps a token to its token ID.
        suffix_to_score: dict[str, float] = {}
        token_to_token_id: dict[str, int] = {}
        self._tokens = []
        for token_id, row in enumerate(vocab):
            assert isinstance(row[0], str), row
            assert isinstance(row[1], (int, float)), row

            token = str(row[0])
            self._tokens.append(token)
            token_to_token_id[token] = token_id

            # Special handling for byte tokens.
            if len(row) > 2 and row[2] == "BYTE":
                assert (
                    len(token) == 6 and token.startswith("<0x") and token.endswith(">")
                ), row[0]
                self._bytes[int(row[0][3:5], 16)] = token_id
                continue

            suffix_to_score[token] = float(row[1])
            # Ensure that all suffixes are included in suffix_to_score.
            for i in range(1, len(token)):
                suffix_to_score[token[i:]] = suffix_to_score.get(token[i:], math.nan)

        # Ensure all byte tokens are set.
        for i in range(256):
            assert self._bytes[i] != 0, f"Byte token for <0x{i:02X}> is not set."

        # List suffixes in lexicographical order of their reversed strings.
        suffixes = list(suffix_to_score.keys())
        suffixes.append("")
        suffixes.sort(key=lambda x: x[::-1])

        # Build suffix_to_id, which is a mapping from a suffix to a suffix ID, and _to_suffix_id,
        # which is a mapping from a piece code to a suffix ID.
        suffix_to_id: dict[str, int] = {}
        num_pieces = 0
        for s in suffixes:
            suffix_to_id[s] = num_pieces
            if s != "":
                self._to_suffix_id[ord(s[0]) << 32 | suffix_to_id[s[1:]]] = np.int32(
                    num_pieces
                )
            num_pieces += 1 + sum(
                s[:i] in suffix_to_score for i in range(1, len(s) + 1)
            )
        assert suffix_to_id[""] == 0, suffix_to_id[""]

        # Build _table, which is a flattened table representing the Trie structure for the Aho-Corasick.
        self._table = np.zeros((num_pieces, 4), dtype=np.int32)
        i = 0
        for suffix in suffixes:
            # Add all prefixes of the suffix to the table.
            for piece_length in range(len(suffix), 0, -1):
                piece = suffix[:piece_length]
                score = suffix_to_score.get(piece, None)
                if score is None:
                    continue
                self._table[i, TABLE_PIECE_LENGTH] = piece_length
                self._table[i, TABLE_TOKEN_ID] = token_to_token_id.get(piece, -1)
                self._table[i, TABLE_SCORE] = (
                    round(score * 1e4) if math.isfinite(score) else INVALID_SCORE
                )
                self._table[i, TABLE_PIECE_ID] = suffix_to_id[piece]
                i += 1

            # Add a sentinel row.
            self._table[i, TABLE_PIECE_LENGTH] = 1
            self._table[i, TABLE_TOKEN_ID] = -1
            self._table[i, TABLE_SCORE] = UNKNOWN_SCORE
            i += 1
        assert i == num_pieces, (i, num_pieces)

    @staticmethod
    @njit
    def _encode(
        to_suffix_id: Dict[types.int64, types.int32],
        table: np.ndarray,
        bytes: np.ndarray,
        data: np.ndarray,
    ) -> np.ndarray:
        # Initialize scores array with a high value and set the score at the end to 0.
        # This array keeps track of the minimum cost (best score) to encode from each position to the end.
        scores = np.full((len(data) + 1,), 2**60, dtype=np.int64)
        scores[-1] = 0

        # Path array to store the best path information.
        # The path array keeps track of token length, token ID, and number of tokens needed to encode.
        path = np.zeros((len(data) + 1, 3), dtype=np.int32)

        # Initialize suffix_id to 0, which represents the root of the Trie.
        suffix_id = 0

        # Process the input data from the end to the beginning.
        for i in range(len(data) - 1, -1, -1):
            c = data[i]

            # Find the next suffix ID by iterating the suffix IDs of prefixes of the current suffix.
            # NOTE: If no suffix ID is found, suffix_id will be set to 0.
            for p in range(suffix_id, len(table)):
                suffix_id = to_suffix_id.get(
                    c << 32 | table[p, TABLE_PIECE_ID], np.int32(0)
                )
                # If a next suffix ID is found or a sentinel row is reached, break the loop.
                if suffix_id > 0 or table[p, TABLE_SCORE] == UNKNOWN_SCORE:
                    break

            # Update the best path to the current position. If multiple paths have the same score,
            # this chooses the longest prefix as the best path (table is sorted in the decreasing
            # order of piece length).
            for p in range(suffix_id, len(table)):
                score = table[p, TABLE_SCORE]
                if score > INVALID_SCORE:
                    piece_length = table[p, TABLE_PIECE_LENGTH]
                    s = scores[i + piece_length] - score
                    if s < scores[i]:
                        scores[i] = s
                        path[i, PATH_TOKEN_LENGTH] = piece_length
                        path[i, PATH_TOKEN_ID] = table[p, TABLE_TOKEN_ID]
                        path[i, PATH_NUM_TOKENS] = (
                            path[i + piece_length, PATH_NUM_TOKENS] + 1
                        )
                        if score == UNKNOWN_SCORE:
                            # Add number of bytes to represent `c` in UTF-8 (minus 1; 1 is already
                            # added above).
                            path[i, PATH_NUM_TOKENS] += (
                                (c >= 0x80) + (c >= 0x800) + (c >= 0x10000)
                            )

                # If it reaches a sentinel row, break the loop.
                if score == UNKNOWN_SCORE:
                    break

        # Decode the best path from the beginning to get the token IDs.
        pos = 0
        token_ids = np.zeros(path[0, PATH_NUM_TOKENS], dtype=np.int32)
        token_pos = 0
        while pos < len(data):
            if path[pos, PATH_TOKEN_ID] >= 0:
                token_ids[token_pos] = path[pos, PATH_TOKEN_ID]
                token_pos += 1
            else:
                # Fall back to byte tokens.
                c = data[pos]
                s = 1 + (c >= 0x80) + (c >= 0x800) + (c >= 0x10000)
                # Add byte tokens representing UTF-8 bytes.
                for i in range(s):
                    b = c if s == 1 else (0xF00 >> s) & 0xFF if i == 0 else 0x80
                    token_ids[token_pos] = bytes[b | ((c >> (s - i - 1) * 6) & 0x3F)]
                    token_pos += 1

            # Ensure that pos should increase by at least 1.
            assert path[pos, PATH_TOKEN_LENGTH] > 0, (pos, path[pos])
            pos += path[pos, PATH_TOKEN_LENGTH]

        return token_ids

    def encode(self, data: str) -> np.ndarray:
        """Encodes a string into a sequence of token IDs."""
        return np.asarray(
            self._encode(
                self._to_suffix_id,
                self._table,
                self._bytes,
                # Convert a string into a numpy array of Unicode code points.
                # NOTE: This skips UTF-32 BOM.
                np.frombuffer(data.encode("utf-32"), dtype=np.int32)[1:],
            )
        )

    def encode_as_tokens(self, data: str) -> list[str]:
        """Encodes a string into a sequence of tokens."""
        return [self._tokens[token_id] for token_id in self.encode(data)]


class Plamo2Tokenizer(PreTrainedTokenizer):  # type: ignore
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    _save_files = [
        "special_tokens_map.json",
        "tokenization_plamo.py",
        "tokenizer.jsonl",
        "tokenizer_config.json",
    ]

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<|plamo:unk|>",
        bos_token: str = "<|plamo:bos|>",
        eos_token: str = "<|plamo:eos|>",
        pad_token: str = "<|plamo:pad|>",
        cls_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        mask_token: Optional[str] = None,
        clean_up_tokenization_spaces: bool = False,
        **kwargs: Any,
    ) -> None:
        """Tokenizer for PLaMo.

        Args:
            vocab_file (str): Vocabrary file path.
            unk_token (str): Unknown token.
            bos_token (str): Beginning of sentence token.
            eos_token (str): End of sentence token.
            pad_token (str): Padding token.
            cls_token (str):
                Classification token, to extract a summary of an input sequence leveraging self-attention along the
                full depth of the model.
            sep_token (str): Separation token, to separate context and query in an input sequence.
            mask_token (str): Mask token, to use when training a model with masked-language modeling.
            clean_up_tokenization_spaces (bool): Whether or not to clean up the tokenization spaces.
            num_threads (int):
                Number of threads. This value will be ignored if one of `PLAMO_TOKENIZER_NUM_THREADS` or
                `RAYON_NUM_THREADS` is set as an environment variable.
        """
        if "add_bos_token" not in kwargs:
            kwargs["add_bos_token"] = False
        if "add_eos_token" not in kwargs:
            kwargs["add_eos_token"] = False
        self.data: list[Any] = [
            json.loads(line) for line in open(vocab_file, "r", encoding="utf-8")
        ]
        self.vocab: dict[str, int] = {v[0]: i for i, v in enumerate(self.data)}
        self.aho_corasick = AhoCorasick()
        self.aho_corasick.build(self.data)
        self.vocab_file = vocab_file
        self.add_bos_token = kwargs["add_bos_token"]
        self.add_eos_token = kwargs["add_eos_token"]

        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    # the functions below are copied from hf transformers LlamaTokenizer's implementation to fix the behaviour of the tokenizer
    # https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/models/llama/tokenization_llama.py

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["aho_corasick"] = None
        return state

    def __setstate__(self, d: dict[str, Any]) -> None:
        self.__dict__ = d
        self.aho_corasick = AhoCorasick()
        self.aho_corasick.build(self.data)

    @property
    def vocab_size(self) -> Any:
        """Returns vocab size"""
        return len(self.data)

    def token_to_score(self, token: str) -> Optional[float]:
        """Returns score of the token"""
        token_id = self.vocab.get(token, None)
        return None if token_id is None else self.data[token_id][1]

    def get_vocab(self) -> dict[str, int]:
        """Returns vocab as a dict"""
        vocab = self.vocab.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        return b"".join(
            [
                bytes([int(t[3:5], 16)]) if t.startswith("<0x") else t.encode("utf-8")
                for t in tokens
            ]
        ).decode("utf-8", errors="replace")

    def _tokenize(self, text: str) -> Any:
        """Returns a tokenized string."""
        return self.aho_corasick.encode_as_tokens(text)

    def _convert_token_to_id(self, token: str) -> Any:
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, 0)

    def _convert_id_to_token(self, index: int) -> Any:
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.data[index][0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return ("",)
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(
            out_vocab_file
        ) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "w") as f:
                for token in self.data:
                    print(json.dumps(token, ensure_ascii=False), file=f)

        return (out_vocab_file,)
