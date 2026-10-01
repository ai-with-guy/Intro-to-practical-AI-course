"""Offline checks of the notebook's real code using a tiny T5 model."""

import html
import json
from pathlib import Path
import unittest

import nbformat
import torch
from transformers import BatchEncoding, T5Config, T5ForConditionalGeneration


NOTEBOOK = Path(__file__).with_name('attention.ipynb')


class SyntheticTokenizer:
    def __call__(self, text, return_tensors):
        length = 257 if text == 'too long' else 5
        return BatchEncoding({
            'input_ids': torch.tensor([[3] * (length - 1) + [2]]),
            'attention_mask': torch.ones(1, length, dtype=torch.long),
        })

    def convert_ids_to_tokens(self, ids):
        return [f'token-{i}' for i in ids]

    def decode(self, ids, skip_special_tokens):
        return ' '.join(self.convert_ids_to_tokens(ids.tolist()))


class AttentionNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(NOTEBOOK, as_version=4)
        torch.manual_seed(0)
        config = T5Config(
            vocab_size=32, d_model=16, d_kv=8, d_ff=32,
            num_layers=2, num_decoder_layers=2, num_heads=2,
            decoder_start_token_id=0, pad_token_id=0, eos_token_id=2,
            attn_implementation='eager',
        )
        cls.model = T5ForConditionalGeneration(config).eval()
        cls.namespace = {
            'torch': torch, 'model': cls.model,
            'tokenizer': SyntheticTokenizer(), 'device': torch.device('cpu'),
        }
        setup = next(c for c in cls.notebook.cells if 'setup' in c.metadata.get('tags', []))
        exec(setup.source, cls.namespace)
        cls.namespace['device'] = torch.device('cpu')
        function = next(c for c in cls.notebook.cells
                        if 'translation-function' in c.metadata.get('tags', []))
        exec(function.source, cls.namespace)
        cls.inspect = staticmethod(cls.namespace['inspect_translation'])
        cls.example = cls.inspect('short method', max_new_tokens=5)

    def test_notebook_valid_and_cells_compile(self):
        nbformat.validate(self.notebook)
        for i, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'{NOTEBOOK.name}:cell-{i}', 'exec')

    def test_decoder_tokens_align_with_queries_not_predictions(self):
        example = self.example
        ids = example['generated_ids'][0]
        self.assertEqual(ids[0].item(), self.model.config.decoder_start_token_id)
        self.assertEqual(example['decoder_tokens'], SyntheticTokenizer().convert_ids_to_tokens(ids[:-1].tolist()))
        self.assertEqual(example['predicted_tokens'], SyntheticTokenizer().convert_ids_to_tokens(ids[1:].tolist()))
        self.assertEqual(len(example['cross_attention']), 2)
        for cross, decoder, encoder in zip(example['cross_attention'],
                                           example['decoder_attention'], example['encoder_attention']):
            self.assertEqual(cross.shape, (1, 2, len(ids) - 1, 5))
            self.assertEqual(encoder.shape, (1, 2, 5, 5))
            torch.testing.assert_close(encoder.sum(-1), torch.ones_like(encoder.sum(-1)))
            torch.testing.assert_close(cross.sum(-1), torch.ones_like(cross.sum(-1)))
            torch.testing.assert_close(decoder.sum(-1), torch.ones_like(decoder.sum(-1)))
            torch.testing.assert_close(decoder.triu(1), torch.zeros_like(decoder))
            # Start token stays visible even though its ID equals the pad ID.
            torch.testing.assert_close(decoder[0, :, 0, 0], torch.ones(2))
            self.assertEqual(cross.device.type, 'cpu')

    def test_input_limits(self):
        with self.assertRaisesRegex(ValueError, '256 source tokens'):
            self.inspect('too long')
        with self.assertRaisesRegex(ValueError, 'positive'):
            self.inspect('short method', max_new_tokens=0)

    def test_shared_bertviz_renders_all_three_attention_types(self):
        example = self.example
        rendered = self.namespace['head_view'](
            **{key: example[key] for key in (
                'encoder_attention', 'decoder_attention', 'cross_attention',
                'encoder_tokens', 'decoder_tokens')},
            heads=[0], html_action='return')
        document = html.unescape(rendered.data)
        payload = document.split('window.BERTVIZ_PARAMS = ', 1)[1].split(';\n', 1)[0]
        params = json.loads(payload)
        self.assertEqual([view['name'] for view in params['attention']], ['Encoder', 'Decoder', 'Cross'])
        self.assertEqual(params['attention'][2]['left_text'], example['decoder_tokens'])
        self.assertEqual(params['attention'][2]['right_text'], example['encoder_tokens'])
        self.assertIn('<iframe', rendered.data)


if __name__ == '__main__':
    unittest.main()
