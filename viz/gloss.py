"""Which tokens of a slice get a translation, and the JSON-ready payload the page's gloss box
reads (the contract with viz/static/gloss.js):

    {"context_ids": [id, ...],                       # prompt-strip token ids, in strip order
     "glosses": {"<token id>": {"text": "都已经",     # the token, whitespace stripped
                 "segments": [{"word": "都", "entries": [{"reading": "dou1", "senses": [...]}]},
                              ...]}}}                # entries [] = no dictionary entry
"""

from jlens.vis import SliceData

from viz.cedict import Cedict, Entry, Segment, segment

LAST_LATIN = 0x024F   # letters above this code point (past Latin Extended-B) count as foreign


def is_foreign(text: str) -> bool:
    """True if `text` has a letter outside the Latin blocks (Han, Kana, Hangul, Cyrillic, ...)."""
    return any(ch.isalpha() and ord(ch) > LAST_LATIN for ch in text)


def token_gloss(cedict: Cedict, text: str) -> dict:
    """The token's stripped text and its dictionary segments."""
    stripped = text.strip()
    return {"text": stripped, "segments": [_segment_json(s) for s in segment(cedict, stripped)]}


def gloss_payload(slice_data: SliceData, cedict: Cedict) -> dict:
    """Glosses for every foreign token the page can show (`vocab_fragment`), keyed by id as a
    string, plus the ids of the prompt strip's tokens (from `ctx_offset` on, as the page shows)."""
    glosses = {str(tid): token_gloss(cedict, text)
               for tid, text in sorted(slice_data.vocab_fragment.items()) if is_foreign(text)}
    context = slice_data.context_token_ids[slice_data.ctx_offset:]
    return {"context_ids": [int(t) for t in context], "glosses": glosses}


def _segment_json(seg: Segment) -> dict:
    return {"word": seg.word, "entries": [_entry_json(e) for e in seg.entries]}


def _entry_json(entry: Entry) -> dict:
    return {"reading": entry.pinyin, "senses": list(entry.senses)}
