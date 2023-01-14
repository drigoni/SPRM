import numpy as np
import spacy
from spellchecker import SpellChecker


def load_boxes_classes(
    file_classes="data/objects_vocab.txt",
    word_embedding=None,
    word_indexer=None,
    do_spellchecker=False,
    do_oov=False,
):
    # read labels
    with open(file_classes, "r") as f:
        data = f.readlines()
    labels = [label.strip() for label in data]

    # correct phrase
    if do_spellchecker:
        print("Spell checking labels...")
        spell: SpellChecker = SpellChecker()
        for i in range(len(labels)):
            label = labels[i]
            label_corrected = spell.correction(label)
            if label_corrected is not None:
                labels[i] = label_corrected

    # correct labels made up of multiple words e.g. "bus stop" and with different
    # variants e.g. "stop sign,stopsign"
    if do_oov and word_indexer is not None and word_embedding is not None:
        print("Correcting oov label embeddings...")
        for i in range(len(labels)):
            label = labels[i]

            if word_indexer.contains(label):
                continue

            label_parts = label.split(",")  # multiple variants
            label_parts_embedding = []

            for label_part in label_parts:
                label_words = label_part.split(" ")  # multiple words
                label_words_embedding = [
                    word_embedding.get_embedding(word)
                    for word in label_words
                    if word_indexer.contains(word)
                ]

                if len(label_words_embedding) > 0:
                    label_part_embedding = np.mean(
                        label_words_embedding, axis=0
                    ).reshape(-1)
                    label_parts_embedding.append(label_part_embedding)

            # we found some words for each label part, then we combine them and
            # add the label to the word embedding along with its new embedding
            if len(label_parts_embedding) > 0:
                label_embedding = np.mean(label_parts_embedding, axis=0).reshape(-1)

                added_idx = word_indexer.add_and_get_index(label)
                word_embedding.vectors = np.insert(
                    word_embedding.vectors, added_idx, label_embedding, axis=0
                )

    return labels


def get_spacy_nlp():
    """
    Returns a `nlp` object with custom rules for "/" and "-" prefixes.
    Resources:
    - [Customizing spaCy’s Tokenizer class](https://spacy.io/usage/linguistic-features#native-tokenizers)
    - [Modifying existing rule sets](https://spacy.io/usage/linguistic-features#native-tokenizer-additions)
    """
    nlp = spacy.load("en_core_web_sm")

    prefixes = nlp.Defaults.prefixes + [r"""/""", r"""-"""]
    prefix_regex = spacy.util.compile_prefix_regex(prefixes)
    nlp.tokenizer.prefix_search = prefix_regex.search

    return nlp


def get_box_relations(boxes, labels, image_width, image_height, *, enabled=False):
    N_RELATIONS = 6
    
    relations = [[1 for i in range(N_RELATIONS)] for j in range(len(boxes))]

    if enabled:
        get_center = lambda x: ((x[0] + x[2]) / 2, (x[1] + x[3]) / 2)

        indexes = [i for i in range(len(boxes))]
        centers = [get_center(box) for box in boxes]

        for label in set(labels):
            indexes_by_label = [i for i in indexes if labels[i] == label]
            centers_by_label = [centers[i] for i in indexes_by_label]

            leftmost = min(centers_by_label, key=lambda x: x[0])[0]    # x
            rightmost = max(centers_by_label, key=lambda x: x[0])[0]   # x
            topmost = min(centers_by_label, key=lambda x: x[1])[1]     # y
            bottommost = max(centers_by_label, key=lambda x: x[1])[1]  # y

            if len(indexes_by_label) > 1:
                for box_index in indexes_by_label:
                    relations[box_index][0] = 1 if centers[box_index][0] == leftmost else 0
                    relations[box_index][1] = 1 if centers[box_index][0] == rightmost else 0
                    relations[box_index][2] = 1 if centers[box_index][0] != leftmost and centers[box_index][0] != rightmost else 0
                    relations[box_index][3] = 1 if centers[box_index][1] == topmost else 0
                    relations[box_index][4] = 1 if centers[box_index][1] == bottommost else 0
                    relations[box_index][5] = 1 if centers[box_index][1] != topmost and centers[box_index][1] != bottommost else 0
    
    return relations


def get_query_locations(query, enabled=False):
    N_LOCATIONS = 6
    
    locations = [[1 for i in range(N_LOCATIONS)] for j in range(len(query))]

    if enabled:
        # locations [left, right, top, bottom]
        for i in range(len(query)):
            noun_phrase = query[i]
            location = [
                1 if "left" in noun_phrase else 0,
                1 if "right" in noun_phrase else 0,
                1 if "center" in noun_phrase else 0,
                1 if "top" in noun_phrase else 0,
                1 if "bottom" in noun_phrase else 0,
                1 if "middle" in noun_phrase else 0,
            ]
            if sum(location) > 0:
                locations[i] = location

    return locations
