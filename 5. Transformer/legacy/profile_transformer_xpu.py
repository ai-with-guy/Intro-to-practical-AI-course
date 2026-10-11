"""Profile actual TinyStories training batches on CPU and XPU/CUDA.

Run: .venv/Scripts/python.exe profile_transformer_xpu.py
Open the emitted JSON in https://ui.perfetto.dev/ (or Chrome tracing).
Downloads/loads TinyStories and the GPT-2 tokenizer, as in the notebook.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from benchmark_transformer_xpu import PAD, VOCAB, Transformer, synchronize


def summarize_trace(path):
    with path.open(encoding='utf-8') as f:
        events = json.load(f)['traceEvents']
    gpu = sorted((e['ts'], e['ts'] + e['dur'], e['name']) for e in events
                 if e.get('ph') == 'X' and e.get('cat') in ('kernel', 'gpu_memcpy')
                 and e.get('dur', 0) > 0)
    if not gpu:
        print('No device events in trace. CPU times alone cannot explain GPU idle time.')
        return
    gaps = []
    busy = 0
    start, end, _ = gpu[0]
    for a, b, name in gpu[1:]:
        if a > end:
            busy += end - start
            gaps.append((a - end, end, name))
            start, end = a, b
        else:
            end = max(end, b)
    busy += end - start
    span = gpu[-1][1] - gpu[0][0]
    print(f'\nDevice events: {len(gpu)}; kernel/memcpy busy: {busy / 1000:.2f} ms '
          f'of {span / 1000:.2f} ms first-to-last-event span ({100 * busy / span:.1f}%).')
    print('Largest gaps between device events (relative to first event; inspect JSON for cause):')
    for duration, at, following in sorted(gaps, reverse=True)[:10]:
        print(f'  +{(at - gpu[0][0]) / 1000:8.2f} ms: '
              f'{duration / 1000:8.2f} ms idle, then {following}')
    print('This is kernel timeline occupancy, NOT hardware EU utilization.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('xpu', 'cuda'), default='xpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=400)
    parser.add_argument('--d-model', type=int, default=23)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=10, help='number of REAL batches to trace')
    parser.add_argument('--loss-layout', choices=('transposed', 'flat'), default='transposed',
                        help='transposed matches notebook; flat uses [batch*tokens, vocab]')
    parser.add_argument('--output', type=Path, default=Path('transformer_xpu_trace.json'))
    args = parser.parse_args()
    if (min(args.batch_size, args.d_model, args.layers, args.steps) < 1 or args.seq_len < 2
            or args.warmup < 0):
        parser.error('sizes and steps must be positive, seq-len >= 2, warmup >= 0')
    activity = ProfilerActivity.XPU if args.device == 'xpu' else ProfilerActivity.CUDA
    if activity not in torch.profiler.supported_activities():
        parser.error(f'{args.device} profiler activity is not supported by this torch build')
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if len(tokenizer) + 1 != VOCAB or tokenizer.eos_token_id != 50256:
        raise RuntimeError('GPT-2 tokenizer IDs differ from the model vocabulary')
    stories = load_dataset('roneneldan/TinyStories')['train'].select_columns('text')

    def collate_fn(examples):
        encoded = tokenizer([example['text'] for example in examples],
                            add_special_tokens=False, truncation=True,
                            max_length=args.seq_len - 1)['input_ids']
        sequences = [ids + [tokenizer.eos_token_id] for ids in encoded]
        width = max(map(len, sequences))
        batch = torch.full((len(sequences), width), PAD, dtype=torch.long)
        for row, ids in enumerate(sequences):
            batch[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        return batch

    loader = iter(DataLoader(stories, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, pin_memory=True, num_workers=0))
    model = Transformer(args.d_model, args.layers, args.seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    print(f'torch={torch.__version__} device={device} dataset=TinyStories '
          f'loss={args.loss_layout} batch_size={args.batch_size} max_len={args.seq_len} '
          f'warmup={args.warmup} traced_batches={args.steps}', flush=True)
    print(f'At maximum length, logits alone: '
          f'{args.batch_size * (args.seq_len - 1) * VOCAB * 4 / 2**20:.1f} MiB', flush=True)

    def step():
        with record_function('step/dataloader_tokenize_pin'):
            host = next(loader)
        with record_function('step/input_copy'):
            batch = host.to(device, non_blocking=True)
        with record_function('step/zero_grad'):
            optimizer.zero_grad()
        with record_function('step/forward'):
            pred = model(batch[:, :-1])
        with record_function('step/cross_entropy'):
            if args.loss_layout == 'flat':
                loss = F.cross_entropy(pred.reshape(-1, VOCAB), batch[:, 1:].reshape(-1),
                                       ignore_index=PAD)
            else:
                loss = F.cross_entropy(pred.transpose(1, 2), batch[:, 1:], ignore_index=PAD)
        with record_function('step/backward'):
            loss.backward()
        with record_function('step/adam'):
            optimizer.step()

    for _ in range(args.warmup):
        step()
    synchronize(device)
    print('Starting profiled training...', flush=True)
    trace_started = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, activity], record_shapes=True) as prof:
        for _ in range(args.steps):
            with record_function('training_step'):
                step()
            prof.step()
        synchronize(device)
    elapsed = time.perf_counter() - trace_started
    print(f'Traced: {args.steps} distinct TinyStories batches in {elapsed:.2f}s '
          f'({elapsed * 1000 / args.steps:.2f} ms/batch including profiler overhead)', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.output))
    print('\nTop CPU self time (may include waits on previously submitted GPU work):')
    print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=18))
    print('Top device operations:')
    print(prof.key_averages().table(sort_by='self_xpu_time_total' if args.device == 'xpu'
                                    else 'self_device_time_total', row_limit=18))
    summarize_trace(args.output)
    print(f'Trace: {args.output.resolve()}')


if __name__ == '__main__':
    main()
