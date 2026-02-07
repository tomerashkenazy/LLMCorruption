from llm_validation import CorruptionClassifier


def test_llm_validation_repetition():
    classifier = CorruptionClassifier(use_llm=False)
    text = "ha ha ha ha ha ha ha"
    result = classifier.classify_response(text)
    assert result["is_corrupted"] is True
    assert result["corruption_type"] in {"REPETITION", "GARBAGE_OUTPUT"}


def test_llm_validation_normal():
    classifier = CorruptionClassifier(use_llm=False)
    text = "This is a short, coherent sentence with normal words."
    result = classifier.classify_response(text)
    assert result["is_corrupted"] is False
