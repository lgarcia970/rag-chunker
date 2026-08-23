from rag_chunker.sentences import split_sentences


def test_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   \n\t\n") == []


def test_single_sentence_no_terminal_punctuation():
    assert split_sentences("just a fragment") == ["just a fragment"]


def test_basic_split():
    text = "First sentence. Second sentence. Third sentence."
    assert split_sentences(text) == [
        "First sentence.",
        "Second sentence.",
        "Third sentence.",
    ]


def test_question_and_exclamation_marks():
    text = "Is this it? Yes it is! Good."
    assert split_sentences(text) == ["Is this it?", "Yes it is!", "Good."]


def test_abbreviation_does_not_split():
    text = "See the docs, e.g. the README, for more."
    assert split_sentences(text) == [text]


def test_title_abbreviation_does_not_split():
    text = "Dr. Chen reviewed the change."
    assert split_sentences(text) == [text]


def test_middle_initial_does_not_split():
    text = "Ask J. Smith about it."
    assert split_sentences(text) == [text]


def test_version_number_does_not_split():
    text = "Upgrade to v1.2.3 before deploying."
    assert split_sentences(text) == [text]


def test_numbered_list_marker_does_not_split():
    text = "1. First step\n2. Second step\n3. Third step"
    assert split_sentences(text) == [text]


def test_trailing_quote_after_terminator():
    text = 'She said "stop." Then left.'
    assert split_sentences(text) == ['She said "stop."', "Then left."]


def test_trailing_paren_after_terminator():
    text = "It works (mostly.) Next."
    assert split_sentences(text) == ["It works (mostly.)", "Next."]


def test_multiple_spaces_between_sentences():
    text = "One.   Two."
    assert split_sentences(text) == ["One.", "Two."]


def test_newline_between_sentences():
    text = "One.\nTwo."
    assert split_sentences(text) == ["One.", "Two."]


def test_no_trailing_whitespace_in_final_sentence():
    text = "Only sentence.   "
    assert split_sentences(text) == ["Only sentence."]


def test_leading_and_trailing_whitespace_stripped():
    text = "  Leading space. Trailing thought  "
    assert split_sentences(text) == ["Leading space.", "Trailing thought"]


def test_abbreviation_at_end_of_text():
    text = "Refer to the appendix etc."
    assert split_sentences(text) == [text]


def test_cjk_terminator_needs_following_space_to_split():
    # the boundary regex requires trailing whitespace or end-of-string, and
    # CJK prose is not space-separated, so a run with no spaces stays whole
    assert split_sentences("第一句。第二句。") == ["第一句。第二句。"]
    assert split_sentences("第一句。 第二句。") == ["第一句。", "第二句。"]
