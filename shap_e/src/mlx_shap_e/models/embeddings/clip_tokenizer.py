from __future__ import annotations

import gzip
import html
import importlib.util
from pathlib import Path
from typing import Iterable

import numpy as np
import regex as re

try:
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None


def default_clip_bpe_path() -> Path:
    spec = importlib.util.find_spec("clip")
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            "Could not locate 'clip' package for tokenizer merges. "
            "Install clip or pass --clip-bpe-path explicitly."
        )
    return Path(spec.origin).resolve().parent / "bpe_simple_vocab_16e6.txt.gz"


def _basic_clean(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class ClipSimpleTokenizer:
    def __init__(self, bpe_path: Path):
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]

        vocab = list(self.byte_encoder.values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> list[int]:
        bpe_tokens: list[int] = []
        text = _whitespace_clean(_basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token_bytes = token.encode("utf-8")
            token_transformed = "".join(self.byte_encoder[b] for b in token_bytes)
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self._bpe(token_transformed).split(" "))
        return bpe_tokens

    def tokenize(self, texts: Iterable[str], context_length: int = 77) -> np.ndarray:
        texts = list(texts)
        sot = self.encoder["<|startoftext|>"]
        eot = self.encoder["<|endoftext|>"]
        tokens = np.zeros((len(texts), context_length), dtype=np.int32)
        for i, text in enumerate(texts):
            row = [sot] + self.encode(text) + [eot]
            if len(row) > context_length:
                row = row[:context_length]
                row[-1] = eot
            tokens[i, : len(row)] = np.asarray(row, dtype=np.int32)
        return tokens
