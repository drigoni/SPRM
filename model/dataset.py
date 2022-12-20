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
