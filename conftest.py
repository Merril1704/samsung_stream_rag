import os
import sys
import pytest
from retrieval.indexer import build_index

sys.path.insert(0, os.path.dirname(__file__))

GOLD_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "eval", "gold_manifest.md")


@pytest.fixture(scope="session")
def indexed_dev_corpus():
    corpus_dir = os.path.join(os.path.dirname(__file__), "corpus", "raw")
    return build_index(corpus_dir)
