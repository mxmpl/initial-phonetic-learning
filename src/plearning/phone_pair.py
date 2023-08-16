from typing import Literal

import pandas as pd


class PhonePair:
    _NO_PAIR_STRING = "Full"

    def __init__(
        self,
        pair: str | None = None,
        language: Literal["Japanese", "English"] = "English",
        *,
        first_phone: str | None = None,
        second_phone: str | None = None,
        reverse_print: bool = False,
    ) -> None:
        self.first: str | None = None
        self.second: str | None = None
        self._language = language
        self.reverse_print = reverse_print

        if pd.isna(pair):
            pair = None
        _verify_phone_pair_input(pair, first_phone, second_phone, language)
        alphabet = JAPANESE_PHONES if language == "Japanese" else AMERICAN_ENGLISH_PHONES
        if pair is not None:
            first_phone, second_phone = pair.replace("[", "").replace("]", "").split("-")
        if (first_phone is not None) and (second_phone is not None):
            first_phone, second_phone = _fix_pair(first_phone, second_phone, language)
            self.first = alphabet[first_phone]
            self.second = alphabet[second_phone]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhonePair) and not isinstance(other, str):
            return NotImplemented
        if isinstance(other, str):
            try:
                other = PhonePair(other, self._language)
            except (KeyError, ValueError):
                return False
        if self.first is None:
            return other.first is None and other.second is None
        return (self.first == other.first) and (self.second == other.second)

    def __repr__(self) -> str:
        return f"PhonePair(first={self.first}, second={self.second}, _language={self._language})"

    def __str__(self) -> str:
        if self.first is None:
            return self._NO_PAIR_STRING
        if self.reverse_print:
            return f"[{self.second}]-[{self.first}]"
        return f"[{self.first}]-[{self.second}]"

    def __hash__(self) -> int:
        return hash(repr(self))

    @property
    def tipa(self) -> str:
        if self.first is None or self.second is None:
            return self._NO_PAIR_STRING
        if self.reverse_print:
            return f"[{TIPA[self.second]}]-[{TIPA[self.first]}]"
        return f"[{TIPA[self.first]}]-[{TIPA[self.second]}]"

    @property
    def ipa(self) -> str:
        return str(self)


def _verify_phone_pair_input(
    pair: str | None, first: str | None, second: str | None, language: Literal["Japanese", "English"]
) -> None:
    if language not in ("Japanese", "English"):
        raise ValueError(f"Invalid language {language}." + " Only 'Japanese' and 'English' supported.")
    if pair is None and first is None and second is None:
        return
    if (pair is None and (first is None or second is None)) or (
        pair is not None and (first is not None or second is not None)
    ):
        raise ValueError("Must provide either the pair or both phones.")


def _fix_pair(first_phone: str, second_phone: str, language: str) -> tuple[str, str]:
    if not first_phone < second_phone:
        first_phone, second_phone = second_phone, first_phone
    if first_phone == "W" and second_phone == "Y" and language == "Japanese":
        return "w", "y"
    if first_phone == "w" and second_phone == "y" and language == "English":
        return "W", "Y"
    return first_phone, second_phone


AMERICAN_ENGLISH_CONSONANTS = {
    # "DX": "?",
    "B": "b",
    "CH": "ʧ",
    "D": "d",
    "DH": "ð",
    "F": "f",
    "G": "g",
    "HH": "h",
    "JH": "ʤ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

AMERICAN_ENGLISH_VOWELS = {
    "IY": "iː",
    "IH": "ɪ",
    "EH": "ɛ",
    "EY": "eɪ",
    "AE": "æ",
    "AA": "ɑː",
    "AW": "aʊ",
    "AY": "aɪ",
    "AH": "ʌ",
    "AO": "ɔː",
    "OY": "ɔɪ",
    "OW": "oʊ",
    "UH": "ʊ",
    "UW": "uː",
    "ER": "ɝ",
}

AMERICAN_ENGLISH_PHONES = {
    **AMERICAN_ENGLISH_CONSONANTS,
    **AMERICAN_ENGLISH_VOWELS,
    "SIL": "SIL",
}

JAPANESE_CONSONANTS = {
    "w": "w",
    "y": "j",
    "m": "m",
    "n": "n",
    "N": "ɴ",
    "d": "d",
    "t": "t",
    "Q+t": "tː",
    "c": "t͡s",
    "c+y": "t͡ɕ",
    "Q+c+y": "t͡ɕː",
    "s": "s",
    "Q+s": "sː",
    "s+y": "ɕ",
    "Q+s+y": "ɕː",
    "z": "z",
    "z+y": "ʑ",
    "F": "ɸ",
    "h": "h",
    "g": "g",
    "k": "k",
    "Q+k": "kː",
    "p": "p",
    "Q+p": "pː",
    "r": "r",
    "b": "b",
    "Q+c": "t͡sː",
}

JAPANESE_CONSONANTS_EXCLUDED = {
    "Q+g": "gː",
    "Q+h": "hː",
    "Q+d": "dː",
    "Q+z": "zː",
    "Q+z+y": "ʑː",
    "Q+F": "ɸː",
    "Q+b": "bː",
}

JAPANESE_VOWELS = {
    "a": "ä",
    "e": "e",
    "i": "i",
    "o": "o",
    "u": "ɯ",
    "a+H": "äː",
    "e+H": "eː",
    "i+H": "iː",
    "o+H": "oː",
    "u+H": "ɯː",
}

JAPANESE_PHONES = {**JAPANESE_CONSONANTS, **JAPANESE_VOWELS}

TIPA = {
    "w": "\\textipa{w}",
    "j": "\\textipa{j}",
    "m": "\\textipa{m}",
    "n": "\\textipa{n}",
    "ɴ": "\\textipa{\\;N}",
    "d": "\\textipa{d}",
    "t": "\\textipa{t}",
    "tː": "\\textipa{t:}",
    "t͡s": "\\textipa{\\t{ts}}",
    "t͡ɕ": "\\textipa{\\t{tC}}",
    "t͡ɕː": "\\textipa{\\t{tC:}}",
    "s": "\\textipa{s}",
    "sː": "\\textipa{s:}",
    "ɕ": "\\textipa{C}",
    "ɕː": "\\textipa{C:}",
    "z": "\\textipa{z}",
    "ʑ": "\\textipa{\\textctz}",
    "ɸ": "\\textipa{F}",
    "h": "\\textipa{h}",
    "g": "\\textipa{g}",
    "k": "\\textipa{k}",
    "kː": "\\textipa{k:}",
    "p": "\\textipa{p}",
    "pː": "\\textipa{p:}",
    "r": "\\textipa{r}",
    "b": "\\textipa{b}",
    "t͡sː": "\\textipa{\\t{ts:}}",
    "ä": "\\textipa{ä}",
    "e": "\\textipa{e}",
    "i": "\\textipa{i}",
    "o": "\\textipa{o}",
    "ɯ": "\\textipa{W}",
    "äː": "\\textipa{ä:}",
    "eː": "\\textipa{e:}",
    "iː": "\\textipa{i:}",
    "oː": "\\textipa{o:}",
    "ɯː": "\\textipa{W:}",
    "ʧ": "\\textipa{\\textteshlig}",
    "ð": "\\textipa{D}",
    "f": "\\textipa{f}",
    "ʤ": "\\textipa{\\textdyoghlig}",
    "l": "\\textipa{l}",
    "ŋ": "\\textipa{N}",
    "ɹ": "\\textipa{\\*r}",
    "ʃ": "\\textipa{S}",
    "θ": "\\textipa{T}",
    "v": "\\textipa{v}",
    "ʒ": "\\textipa{Z}",
    "ɪ": "\\textipa{I}",
    "ɛ": "\\textipa{E}",
    "eɪ": "\\textipa{eI}",
    "æ": "\\textipa{\\ae}",
    "ɑː": "\\textipa{A:}",
    "aʊ": "\\textipa{aU}",
    "aɪ": "\\textipa{aI}",
    "ʌ": "\\textipa{\\textturnv}",
    "ɔː": "\\textipa{O:}",
    "ɔɪ": "\\textipa{OI}",
    "oʊ": "\\textipa{oU}",
    "ʊ": "\\textipa{U}",
    "uː": "\\textipa{u:}",
    "ɝ": "\\textipa{\\textrhookrevepsilon}",
    "SIL": "SIL",
}


AMERICAN_ENGLISH_SONORITY = {
    "fricative": ["F", "V", "S", "Z", "ZH", "SH", "HH", "DH", "TH"],
    "affricate": ["JH", "CH"],
    "plosive": ["P", "B", "D", "T", "K", "G"],
    "approximant": ["W", "Y", "R", "L"],
    "nasal": ["M", "N", "NG"],
    "vowel": list(AMERICAN_ENGLISH_VOWELS.keys()),
}
