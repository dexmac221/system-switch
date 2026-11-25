"""The held-out test set of the first paper repeats questions: of its 600 command questions 440 are distinct
texts (state + question), of its 300 weapon questions 242. Helpers shared by the analysis scripts."""
import json


def text_key(row):
    return json.dumps([row["state"], row["question"]], sort_keys=True)


def first_occurrence(test):
    """index of every question -> index of the first question with the same text"""
    first, out = {}, {}
    for i, r in enumerate(test):
        out[i] = first.setdefault(text_key(r), i)
    return out


def keep_indices(test, distinct):
    """all indices, or one index per distinct text (its first occurrence)"""
    return set(first_occurrence(test).values()) if distinct else set(range(len(test)))
