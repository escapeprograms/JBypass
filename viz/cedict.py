"""CC-CEDICT (Chinese-English dictionary): parse its lines into an index, split text into
known words by longest match, and read the gzipped file. Standard library only.

The file is downloaded by hand from CEDICT_PAGE (the site forbids scripted access); see viz/NOTICE."""

import gzip
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

CEDICT_PAGE = "https://www.mdbg.net/chinese/dictionary?page=cc-cedict"
CEDICT_FILE = "cedict_1_0_ts_utf-8_mdbg.txt.gz"   # the name MDBG serves the download under

# v1: 繁 简 [pin1 yin1] /sense 1/sense 2/    v2: 繁 简 [[pin1yin1]] /gloss; gloss/sense/
_LINE = re.compile(r"(\S+) (\S+) \[\[?([^\]]*)\]\]? /(.*)/")


@dataclass(frozen=True)
class Entry:
    traditional: str
    simplified: str
    pinyin: str                 # verbatim, tone numbers (v1 spaced, v2 joined)
    senses: tuple[str, ...]     # the "/"-separated senses; ";" kept inside a sense


@dataclass(frozen=True)
class Segment:
    word: str
    entries: tuple[Entry, ...]  # () = no dictionary entry


@dataclass(frozen=True)
class Cedict:
    index: dict[str, tuple[Entry, ...]]   # traditional and simplified headword -> its entries
    max_len: int                          # longest headword, in characters
    n_entries: int                        # parsed lines


def parse_line(line: str) -> Entry:
    """One dictionary line (either bracket form) as an Entry; ValueError if it is not one."""
    match = _LINE.fullmatch(line.strip())
    if match is None:
        raise ValueError(f"not a CC-CEDICT entry: {line.rstrip()!r}")
    traditional, simplified, pinyin, senses = match.groups()
    return Entry(traditional, simplified, pinyin, tuple(senses.split("/")))


def parse_cedict(lines: Iterable[str]) -> Cedict:
    """Index every entry under its traditional and simplified forms. Blank and `#` lines are
    skipped; any other line that does not parse raises with its (1-based) line number."""
    entries = [_parse_numbered(number, line) for number, line in enumerate(lines, start=1)
               if _is_entry_line(line)]
    if not entries:
        raise ValueError("no CC-CEDICT entries found")
    index = _build_index(entries)
    return Cedict(index=index, max_len=max(map(len, index)), n_entries=len(entries))


def segment(cedict: Cedict, text: str) -> list[Segment]:
    """Split `text` into headwords, longest match first at each position. Characters with no
    match merge into one Segment with no entries; whitespace is dropped and ends such a run."""
    segments: list[Segment] = []
    run, i = "", 0
    while i < len(text):
        word = _longest_at(cedict, text, i)
        if word is None and not text[i].isspace():
            run += text[i]
            i += 1
            continue
        if run:
            segments.append(Segment(run, ()))
            run = ""
        if word is not None:
            segments.append(Segment(word, cedict.index[word]))
        i += len(word) if word is not None else 1
    if run:
        segments.append(Segment(run, ()))
    return segments


def load_cedict(path: Path) -> Cedict:
    """Read the gzipped CC-CEDICT file at `path` (the only file access in this module)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"CC-CEDICT not found at {path}. Download {CEDICT_FILE} from {CEDICT_PAGE} "
            f"(download it in a browser: the site prohibits scripted access) and save it there, "
            f"or set `dictionary_file: null` in configs/viz.yaml to turn translations off.")
    with gzip.open(path, "rt", encoding="utf-8-sig") as lines:
        return parse_cedict(lines)


def _is_entry_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _parse_numbered(number: int, line: str) -> Entry:
    try:
        return parse_line(line)
    except ValueError as error:
        raise ValueError(f"CC-CEDICT line {number}: {error}") from error


def _build_index(entries: list[Entry]) -> dict[str, tuple[Entry, ...]]:
    """Headword -> entries in file order, proper nouns (capitalised pinyin) stably moved last."""
    by_word: dict[str, list[Entry]] = {}
    for entry in entries:
        for word in dict.fromkeys((entry.traditional, entry.simplified)):
            by_word.setdefault(word, []).append(entry)
    return {word: tuple(sorted(found, key=_is_proper_noun)) for word, found in by_word.items()}


def _is_proper_noun(entry: Entry) -> bool:
    return entry.pinyin[:1].isupper()


def _longest_at(cedict: Cedict, text: str, start: int) -> str | None:
    """The longest headword starting at `start`, or None."""
    for end in range(min(len(text), start + cedict.max_len), start, -1):
        if text[start:end] in cedict.index:
            return text[start:end]
    return None
