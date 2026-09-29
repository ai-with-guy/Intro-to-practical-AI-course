"""Regression checks for the score-shaped, True-means-excluded mask contract."""

import unittest

import torch

from attention_math_tests import run_attention_math_tests


def masked_attention(q, k, v, scale, mask=None):
    scores = (q @ k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    return scores.softmax(dim=-1) @ v


class AttentionMathTests(unittest.TestCase):
    def test_score_shaped_mask_passes_padding_and_causal_cases(self):
        self.assertTrue(run_attention_math_tests(masked_attention))

    def test_padded_keys_still_must_be_excluded(self):
        def ignores_mask(q, k, v, scale, mask=None):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)

        self.assertFalse(run_attention_math_tests(ignores_mask))

    def test_masked_attention_weights_are_zero(self):
        q = k = torch.zeros(4, 4)
        v = torch.eye(4)
        mask = torch.ones(4, 4, dtype=torch.bool).triu(diagonal=1)
        weights = masked_attention(q, k, v, 1.0, mask)
        self.assertTrue(torch.equal(weights[mask], torch.zeros_like(weights[mask])))


if __name__ == "__main__":
    unittest.main()
