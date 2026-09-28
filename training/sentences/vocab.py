"""Gloss vocabulary (from Isharah SI train; class id = index + 1, blank = 0) and the lookup
from a recognised gloss sequence to its written Arabic sentence."""


def build_vocab(rows, split=("SI", "train")):
    train = [r for r in rows if r["splits"].get(split[0]) == split[1]]
    glosses = sorted({g for r in train for g in r["gloss"].split()})
    known = set(glosses)
    lookup = {r["gloss"]: r["text"] for r in rows if all(g in known for g in r["gloss"].split())}
    return {"version": 1, "fps": 15, "blank": 0, "glosses": glosses, "lookup": lookup}


def encode(vocab, gloss):
    index = {g: i + 1 for i, g in enumerate(vocab["glosses"])}
    return [index[g] for g in gloss.split() if g in index]


def decode_text(vocab, ids):
    key = " ".join(vocab["glosses"][i - 1] for i in ids)
    return vocab["lookup"].get(key, key)
