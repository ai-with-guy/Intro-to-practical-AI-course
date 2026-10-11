"""Offline DETR attention checks with a tiny, randomly initialized detector."""

import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor, ResNetConfig

from detr_attention import (
    inspect_detection, detection_attention_map, plot_detection_attention,
    detection_attention_view,
)


class DetrAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        config = DetrConfig(
            backbone_config=ResNetConfig(
                embedding_size=8, hidden_sizes=[8, 16, 24, 32],
                depths=[1, 1, 1, 1], out_features=['stage4']),
            d_model=32, encoder_layers=1, decoder_layers=2,
            encoder_attention_heads=2, decoder_attention_heads=2,
            encoder_ffn_dim=64, decoder_ffn_dim=64, num_queries=5,
            num_labels=3, attn_implementation='eager',
        )
        cls.model = DetrForObjectDetection(config).eval()
        cls.processor = DetrImageProcessor(size={'shortest_edge': 64, 'longest_edge': 96})
        cls.image = Image.new('RGB', (96, 64), color='gray')
        cls.example = inspect_detection(
            cls.image, model=cls.model, processor=cls.processor, threshold=0)

    def test_grid_and_attention_normalization(self):
        example = self.example
        self.assertEqual(example['grid_shape'], (2, 3))
        self.assertEqual(len(example['cross_attention']), 2)
        for layer in example['cross_attention']:
            self.assertEqual(layer.shape, (1, 2, 5, 6))
            self.assertEqual(layer.device.type, 'cpu')
            torch.testing.assert_close(layer.sum(-1), torch.ones(1, 2, 5))
        grid = detection_attention_map(example, detection=2, head=1)
        torch.testing.assert_close(grid.flatten(), example['cross_attention'][-1][0, 1, 2])
        backbone = self.model.model.backbone
        self.assertFalse(getattr(backbone, 'conv_encoder', backbone)._forward_hooks)

    def test_postprocessing_matches_processor_and_original_query_ids(self):
        with torch.inference_mode():
            outputs = self.model(**self.processor(images=self.image, return_tensors='pt'))
        result = self.processor.post_process_object_detection(
            outputs, threshold=0, target_sizes=torch.tensor([[64, 96]]))[0]
        for key in ('boxes', 'scores', 'labels'):
            torch.testing.assert_close(self.example[key], result[key])
        # Deliberately non-consecutive retained IDs: never use detection index as query ID.
        filtered = dict(self.example, query_ids=torch.tensor([4, 1]))
        grid = detection_attention_map(filtered, detection=0)
        expected = self.example['cross_attention'][-1][0, :, 4].mean(0)
        torch.testing.assert_close(grid.flatten(), expected)

    def test_empty_detections_and_threshold_validation(self):
        empty = inspect_detection(self.image, model=self.model, processor=self.processor, threshold=1)
        self.assertEqual(empty['boxes'].shape, (0, 4))
        with patch('builtins.print') as message:
            self.assertIsNone(detection_attention_view(empty))
            message.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'Choose a detection'):
            detection_attention_map(empty)
        for threshold in (-0.1, 1.1):
            with self.assertRaisesRegex(ValueError, 'threshold'):
                inspect_detection(self.image, model=self.model, processor=self.processor, threshold=threshold)

    def test_plot_has_boxes_and_raw_spatial_map(self):
        fig = plot_detection_attention(self.example, detection=3, head=0)
        self.assertEqual(len(fig.axes[0].patches), 5)
        self.assertEqual(len(fig.axes[1].patches), 1)
        self.assertEqual(fig.axes[1].images[-1].get_array().shape, (2, 3))
        self.assertIn('Query 3', fig.axes[1].get_title())
        plt.close(fig)

    def test_hook_removed_after_model_failure(self):
        with patch.object(self.model, 'forward', side_effect=RuntimeError('test failure')):
            with self.assertRaisesRegex(RuntimeError, 'test failure'):
                inspect_detection(self.image, model=self.model, processor=self.processor)
        backbone = self.model.model.backbone
        self.assertFalse(getattr(backbone, 'conv_encoder', backbone)._forward_hooks)

    def test_widget_controls_use_recorded_attention(self):
        with patch('IPython.display.display'), patch('matplotlib.pyplot.show'):
            viewer = detection_attention_view(self.example)
            box_control, head_control = viewer.children[0].children
            layer_control, opacity_control = viewer.children[1].children
            self.assertEqual(len(box_control.options), 5)
            self.assertEqual(head_control.value, None)
            self.assertEqual(layer_control.value, 1)
            self.assertEqual(opacity_control.value, 0.6)
            box_control.value = 4
            head_control.value = 1
            layer_control.value = 0
            viewer.close()

    def test_widget_starting_selections(self):
        with patch('IPython.display.display'), patch('matplotlib.pyplot.show'):
            viewer = detection_attention_view(
                self.example, detection=3, layer=0, head=1, overlay=0.35)
            box_control, head_control = viewer.children[0].children
            layer_control, opacity_control = viewer.children[1].children
            self.assertEqual(box_control.value, 3)
            self.assertEqual(head_control.value, 1)
            self.assertEqual(layer_control.value, 0)
            self.assertEqual(opacity_control.value, 0.35)
            self.assertIn('Box 3, q3', box_control.label)
            viewer.close()

    def test_invalid_starting_selections(self):
        for selection in ({'detection': 5}, {'layer': 2}, {'layer': -3},
                          {'head': 2}, {'head': -1}, {'overlay': 1.1}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                detection_attention_view(self.example, **selection)


if __name__ == '__main__':
    unittest.main()
