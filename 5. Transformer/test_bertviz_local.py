"""Check the attention data delivered to the bundled BertViz frontend."""

import html
import json
import unittest

import numpy as np
import torch

from bertviz_local import head_view


def view_params(**kwargs):
    rendered = head_view(html_action='return', **kwargs)
    document = html.unescape(rendered.data)
    payload = document.split('window.BERTVIZ_PARAMS = ', 1)[1].split(';\n', 1)[0]
    return json.loads(payload)


class CustomAttentionViewTests(unittest.TestCase):
    def setUp(self):
        self.weights = torch.tensor([[0.75, 0.25], [0.1, 0.9]])
        self.tokens = ['hello', 'world']

    def test_single_head_formats_preserve_weights_in_frontend(self):
        for attention in (self.weights.requires_grad_(), self.weights.detach().numpy(),
                          self.weights.tolist(), self.weights.unsqueeze(0)):
            with self.subTest(input_type=type(attention).__name__):
                params = view_params(attention=attention, tokens=self.tokens)
                self.assertEqual(params['include_layers'], [0])
                self.assertEqual(params['attention'][0]['attn'], [[self.weights.tolist()]])
                self.assertEqual(params['attention'][0]['left_text'], self.tokens)

    def test_layer_filter_works_for_custom_and_original_formats(self):
        second = self.weights.flip(-1)
        formats = ([self.weights, second],
                   [self.weights.numpy(), second.numpy()],
                   [self.weights.tolist(), second.tolist()],
                   (self.weights[None, None], second[None, None]))
        for attention in formats:
            with self.subTest(input_type=type(attention).__name__):
                params = view_params(attention=attention, tokens=self.tokens,
                                     include_layers=[1], layer=1, heads=[0])
                self.assertEqual(params['include_layers'], [1])
                self.assertEqual(params['attention'][0]['attn'], [[second.tolist()]])

    def test_multi_head_tensor_matches_original_batched_input(self):
        heads = torch.stack([self.weights, self.weights.flip(-1)])
        custom = view_params(attention=heads, tokens=self.tokens)
        original = view_params(attention=(heads.unsqueeze(0),), tokens=self.tokens)
        self.assertEqual(custom['attention'], original['attention'])

    def test_sentence_pair_slices_single_head_attention(self):
        params = view_params(attention=self.weights, tokens=self.tokens, sentence_b_start=1)
        self.assertEqual(len(params['attention']), 5)
        self.assertEqual(params['attention'][3]['attn'], [[[[self.weights[0, 1].item()]]]])
        self.assertEqual(params['attention'][3]['left_text'], ['hello'])
        self.assertEqual(params['attention'][3]['right_text'], ['world'])

    def test_rectangular_custom_cross_attention(self):
        weights = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
        params = view_params(cross_attention=weights,
                             decoder_tokens=self.tokens, encoder_tokens=['a', 'b', 'c'])
        self.assertEqual(params['attention'][0]['attn'], [[weights.tolist()]])
        self.assertEqual(params['attention'][0]['left_text'], self.tokens)
        self.assertEqual(params['attention'][0]['right_text'], ['a', 'b', 'c'])

    def test_invalid_inputs_raise_useful_errors(self):
        cases = [
            ({'attention': torch.zeros(2, 1, 2, 2)}, 'batch size'),
            ({'attention': torch.zeros(2)}, 'shape'),
            ({'attention': []}, 'non-empty'),
            ({'attention': self.weights, 'include_layers': []}, 'at least one'),
            ({'attention': self.weights, 'include_layers': [1]}, 'Layer indices'),
            ({'attention': [self.weights, torch.zeros(3, 3)]}, 'same number'),
            ({'attention': torch.zeros(0, 0)}, 'at least one'),
            ({'attention': torch.zeros(3, 3)}, 'number of tokens'),
        ]
        for arguments, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    view_params(tokens=self.tokens, **arguments)


if __name__ == '__main__':
    unittest.main()
