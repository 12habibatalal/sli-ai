from training.sentences.arabsign import norm


def test_norm_matches_isharah_spelling():
    assert norm("كلمة") == "كلمه"
    assert norm("أول") == "اول" and norm("إسلام") == "اسلام" and norm("آية") == "ايه"
    assert norm("على") == "علي"
    assert norm("الْحَمْد") == "الحمد"
