"""Stress dictionary lookup, after lang-uk/ukrainian-word-stress (MIT) without its Stanza tagger."""

from __future__ import annotations

from pathlib import Path

import marisa_trie

_POS_SEP = b"\xfe"
_REC_SEP = b"\xff"


class StressDict:
    def __init__(self, path: Path):
        self._trie = marisa_trie.BytesTrie()
        self._trie.load(str(path))

    def accent_positions(self, word: str) -> list[int]:
        """Offsets right after each stressed vowel; ambiguous words take the first record."""
        for candidate in (word, word.lower(), word.title()):
            if candidate in self._trie:
                value = self._trie[candidate][0]
                break
        else:
            return []
        if _REC_SEP not in value:
            return list(value)
        for record in value.split(_REC_SEP):
            if record:
                return list(record.partition(_POS_SEP)[0])
        return []
