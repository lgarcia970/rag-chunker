import math

from rag_chunker.tokens import (
    CHARS_PER_WORD_TOKEN,
    estimate_tokens,
    fits_budget,
)


def test_empty_input():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_short_word_costs_one_token():
    assert estimate_tokens("cat") == 1


def test_long_word_scales_by_length():
    word = "a" * 40
    assert estimate_tokens(word) == math.ceil(len(word) / CHARS_PER_WORD_TOKEN)


def test_apostrophe_word_counts_as_single_word():
    assert estimate_tokens("don't") == 2
    assert estimate_tokens("can’t") == 2


def test_digit_run_uses_number_rate():
    assert estimate_tokens("123456789") == 3


def test_short_digit_run_costs_one_token():
    assert estimate_tokens("7") == 1


def test_cjk_characters_cost_one_token_each():
    assert estimate_tokens("第一句") == 3


def test_newlines_are_cheap_and_merge_into_one_run():
    assert estimate_tokens("\n\n\n") == 2


def test_other_symbols_use_symbol_rate():
    assert estimate_tokens("!!!") == 2


def test_spaces_are_free():
    assert estimate_tokens("cat dog") == estimate_tokens("cat   dog")
    assert estimate_tokens("   ") == 0


def test_mixed_text_sums_each_kind():
    text = "hi 42!"
    expected = (
        max(1.0, 2 / CHARS_PER_WORD_TOKEN)
        + max(1.0, 2 / 3.0)
        + 1 * 0.6
    )
    assert estimate_tokens(text) == math.ceil(expected)


def test_result_is_int_and_non_negative():
    result = estimate_tokens("some ordinary sentence with 123 numbers.")
    assert isinstance(result, int)
    assert result >= 0


def test_fits_budget_true_when_under_or_equal():
    text = "short text"
    exact = estimate_tokens(text)
    assert fits_budget(text, exact) is True
    assert fits_budget(text, exact + 1) is True


def test_fits_budget_false_when_over():
    text = "short text"
    exact = estimate_tokens(text)
    assert fits_budget(text, exact - 1) is False


def test_fits_budget_empty_text_always_fits():
    assert fits_budget("", 0) is True
