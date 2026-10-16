"""The scores every decision rests on (tiny_resume_parser/metrics.py). Each test states the rule the protocol fixes:
lenient forgives case, spacing and punctuation but never a letter or digit; content forgives split, merged or
re-placed text within a section; repair keeps only complete values. Run: pytest, or python tests/test_scoring.py"""

from collections import Counter

from tiny_resume_parser.metrics import content, f1, lenient, pairs, parse, repair, score


def matched(pred: Counter, gold: Counter) -> int:
    return sum((pred & gold).values())


def test_lenient_ignores_case_spacing_punctuation():
    gold = Counter({("basics.phones", "+33 6 12-34-56-78"): 1, ("basics.name", "Jean-Luc Picard"): 1})
    pred = Counter({("basics.phones", "+33612345678"): 1, ("basics.name", "JEAN LUC PICARD"): 1})
    assert matched(lenient(pred), lenient(gold)) == 2


def test_lenient_counts_any_digit_or_letter_difference():
    # A date with an invented day, or one misread letter, is a wrong value for an applicant-tracking import.
    gold = Counter({("work.startDate", "2021-04"): 1, ("basics.name", "Picard"): 1})
    pred = Counter({("work.startDate", "2021-04-21"): 1, ("basics.name", "Picart"): 1})
    assert matched(lenient(pred), lenient(gold)) == 0


def test_content_accepts_split_and_other_field_within_a_section():
    # The teacher itself splits and places text inconsistently; content F1 measures whether the words were read.
    gold = pairs({"skills": [{"name": "Python, SQL"}]})
    pred = pairs({"skills": [{"name": "Python"}, {"keywords": ["SQL"]}]})
    assert matched(content(pred), content(gold)) == 2


def test_content_does_not_accept_text_in_another_section():
    gold = pairs({"skills": [{"name": "Python"}]})
    pred = pairs({"interests": [{"name": "Python"}]})
    assert matched(content(pred), content(gold)) == 0


def test_repair_keeps_complete_values_of_a_cut_off_answer():
    answer = '{"basics": {"name": "Ada Lovelace", "emails": ["ada@example.com"]}, "skills": [{"name": "Math'
    # The unfinished "skills" key is dropped whole: a half-written value must not be scored.
    assert repair(answer) == {"basics": {"name": "Ada Lovelace", "emails": ["ada@example.com"]}}


def test_repair_drops_repeated_list_items():
    # A repetition loop writes the same entry until the token limit; repair must not count it more than once.
    answer = '{"skills": [{"name": "SQL"}, {"name": "SQL"}, {"name": "SQL"}]}'
    assert repair(answer) == {"skills": [{"name": "SQL"}]}


def test_f1_of_empty_prediction_is_zero_not_an_error():
    assert f1(0, 0, 10) == (0.0, 0.0, 0.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_answer_nested_past_the_recursion_limit_scores_as_unparsed(monkeypatch):
    # A looping model can write thousands of "[": on Python 3.11 json.loads then raises RecursionError, and one such
    # answer stopped the scoring of a whole epoch. Raised directly here, since Python 3.14 parses deep nesting.
    import tiny_resume_parser.metrics as m

    def too_deep(*a, **k):
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")

    monkeypatch.setattr(m.json, "loads", too_deep)
    deep = '{"skills": [[[[1]]]]}'
    assert score(deep, {})["status"] == "unparsed"
    assert repair(deep) is None
    assert parse(deep, fix=True) == {}
