"""Run with: uv run python -m unittest test_benchmark_utils"""

import unittest
from unittest.mock import patch

import gradio as gr
from benchmark_utils import build_benchmark_ui, parse_metrics, plot_results


SUMMARY = '[ INFO ]    Median: 2.57 ms\n[ INFO ] Throughput: 374.53 FPS\n'


class BenchmarkTests(unittest.TestCase):
    def test_summary_metrics(self):
        self.assertEqual(parse_metrics(SUMMARY), (2.57, 374.53))
        with self.assertRaises(ValueError):
            parse_metrics('[ INFO ] First inference took 123 ms')

    def test_plot_filters_model_and_goal(self):
        rows = [['Face detection', 'CPU', 'latency', 2, 500],
                ['Face detection', 'GPU', 'throughput', 3, 900],
                ['Face embedding', 'CPU', 'latency', 1, 1000]]
        figure = plot_results(rows, 'Face detection', 'latency')
        self.assertEqual(list(figure.axes[0].collections[0].get_offsets()[:, 1]), [2])

    def test_ui_run_replace_failure_and_clear(self):
        demo = build_benchmark_ui(['CPU', 'GPU.0', 'AUTO'], 'Latency')
        callbacks = {fn.fn.__name__: fn.fn for fn in demo.fns.values()}
        measure = callbacks['measure']
        with patch('benchmark_utils.run_benchmark', return_value=(0, SUMMARY, 'benchmark_app -m model.xml')):
            first = measure('Face detection', 'FP16', 'CPU', [])
            second = measure('Face detection', 'FP16', 'CPU', first[0])
        self.assertEqual(len(second[0]), 1)
        self.assertEqual(second[0][0][3:], [2.57, 374.53])
        self.assertIn('benchmark_app', second[2])
        with patch('benchmark_utils.run_benchmark', return_value=(1, 'Device unavailable', 'command')):
            with self.assertRaises(gr.Error):
                measure('Face detection', 'FP16', 'GPU.0', second[0])
        self.assertEqual(len(second[0]), 1)
        self.assertEqual(callbacks['clear_results']('Face detection')[0], [])
        demo.close()

    def test_no_devices(self):
        for goal in ['Latency', 'Throughput']:
            demo = build_benchmark_ui([], goal)
            demo.close()


if __name__ == '__main__':
    unittest.main()
