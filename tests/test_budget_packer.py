"""Tests for the greedy budget packing utility."""

from ctxforge.utils.budget_packer import budget_pack


class TestBudgetPack:

    def test_empty_items(self):
        assert budget_pack([], budget=100, text_fn=str) == []

    def test_zero_budget(self):
        assert budget_pack(["hello world"], budget=0, text_fn=str) == []

    def test_all_fit(self):
        items = ["a b", "c d", "e f"]
        result = budget_pack(items, budget=100, text_fn=str)
        assert result == items

    def test_greedy_packing(self):
        items = ["a b c", "d e", "f"]
        # budget=4 words: first item costs 3, second costs 2 -> only first fits
        result = budget_pack(items, budget=4, text_fn=str)
        assert result == ["a b c", "f"]

    def test_single_item_exceeds_budget(self):
        items = ["a b c d e"]
        result = budget_pack(items, budget=2, text_fn=str)
        assert result == []

    def test_preserves_order(self):
        items = ["short", "a b c d e f g h i j"]
        result = budget_pack(items, budget=5, text_fn=str)
        assert result == ["short"]

    def test_custom_token_fn(self):
        # Custom token function: each character is 1 token
        items = ["ab", "cde", "f"]
        result = budget_pack(items, budget=4, text_fn=str, token_fn=len)
        assert result == ["ab", "f"]

    def test_exact_budget(self):
        items = ["a b", "c d"]
        result = budget_pack(items, budget=4, text_fn=str)
        assert result == ["a b", "c d"]

    def test_negative_budget(self):
        assert budget_pack(["hello"], budget=-1, text_fn=str) == []
