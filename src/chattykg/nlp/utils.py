#!./venv python
# -*- coding: utf-8 -*-

from string import punctuation
from nltk.corpus import wordnet as wn


nltk_POS_map = {
    "VB": wn.VERB,
    "VBD": wn.VERB,
    "VBN": wn.VERB,
    "VBP": wn.VERB,
    "VBZ": wn.VERB,
    "VBG": wn.VERB,
    "JJ": wn.ADJ,
    "NN": wn.NOUN,
    "NNS": wn.NOUN,
    "RB": wn.ADV,
}
table = str.maketrans("", "", punctuation)


def traverse_tree(subtree):
    positions = list()
    positions.append(subtree["spans"][0]["start"])
    if "children" not in subtree:
        return positions

    for child in subtree["children"]:
        ps = traverse_tree(child)
        positions.extend(ps)
    else:
        return positions


def remove_duplicates(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
