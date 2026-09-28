from training.sentences import vocab as V

ROWS = [
    {"gloss": "سوال هو", "text": "من هو", "splits": {"SI": "train"}},
    {"gloss": "هو معلم لغه اشاره", "text": "هو مدرس لغه اشاره", "splits": {"SI": "train"}},
    {"gloss": "هو جديد", "text": "هو جديد", "splits": {"SI": "dev"}},
]


def test_vocab_from_train_only_and_blank_zero():
    v = V.build_vocab(ROWS)
    assert v["blank"] == 0 and "جديد" not in v["glosses"]
    assert V.encode(v, "هو معلم") == [v["glosses"].index("هو") + 1, v["glosses"].index("معلم") + 1]


def test_unknown_gloss_is_dropped_and_text_lookup():
    v = V.build_vocab(ROWS)
    assert V.encode(v, "هو جديد") == [v["glosses"].index("هو") + 1]
    assert V.decode_text(v, V.encode(v, "سوال هو")) == "من هو"
    assert V.decode_text(v, V.encode(v, "هو سوال")) == "هو سوال"  # unknown sentence: glosses joined
