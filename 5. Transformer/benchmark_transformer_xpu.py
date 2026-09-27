"""Isolate copy, transformer operations, and training-step latency.

Run from this directory: .venv/Scripts/python.exe benchmark_transformer_xpu.py
Use --device cuda on a CUDA machine for a like-for-like comparison.
This uses synthetic IDs; it does not download TinyStories or tokenize text.
"""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F
from torch import nn


VOCAB = 50258
PAD = 50257


class Attention(nn.Module):
    def __init__(self, d, max_len):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)))
        self.scale = d ** -0.5

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~self.mask[:x.shape[1], :x.shape[1]], float('-inf'))
        return torch.softmax(scores, dim=-1) @ v


class Block(nn.Module):
    def __init__(self, d, max_len):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attention = Attention(d, max_len)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, d, layers, max_len):
        super().__init__()
        self.pos = nn.Embedding(max_len, d)
        self.tok = nn.Embedding(VOCAB, d)
        self.dropout = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*(Block(d, max_len) for _ in range(layers)))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.dropout(self.tok(ids) + self.pos(positions))
        return self.head(self.norm(self.blocks(x)))


def synchronize(device):
    if device.type == 'xpu':
        torch.xpu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def measure(name, fn, device, warmup, repeats):
    # Each iteration begins idle. The difference between submit and wait shows
    # whether Python is blocked or the queued device work is still outstanding.
    samples = []
    for i in range(warmup + repeats):
        synchronize(device)
        start = time.perf_counter_ns()
        result = fn()
        submitted = time.perf_counter_ns()
        synchronize(device)
        finished = time.perf_counter_ns()
        if i >= warmup:
            samples.append(((submitted - start) / 1e6,
                            (finished - submitted) / 1e6,
                            (finished - start) / 1e6))
        del result
    med = [statistics.median(row[j] for row in samples) for j in range(3)]
    p95 = sorted(row[2] for row in samples)[int(0.95 * (len(samples) - 1))]
    print(f'{name:<31} {med[0]:>10.3f} {med[1]:>10.3f} {med[2]:>10.3f} {p95:>10.3f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('xpu', 'cuda', 'cpu'), default='xpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=128)
    parser.add_argument('--d-model', type=int, default=23)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    args = parser.parse_args()
    if min(args.batch_size, args.d_model, args.layers, args.repeats) < 1 or args.seq_len < 2 or args.warmup < 0:
        parser.error('sizes and repeats must be positive, seq-len >= 2, warmup >= 0')
    device = torch.device(args.device)
    if args.device == 'xpu' and not torch.xpu.is_available():
        parser.error('XPU is not available')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is not available')

    b, s, d = args.batch_size, args.seq_len, args.d_model
    print(f'torch={torch.__version__} device={device} batch={b} seq={s} d={d} layers={args.layers}')
    if device.type == 'xpu':
        print(torch.xpu.get_device_properties(0))
    print(f'Logits alone: {b * (s - 1) * VOCAB * 4 / 2**20:.1f} MiB (training needs much more).')
    print(f'{"operation":<31} {"submit ms":>10} {"wait ms":>10} {"total ms":>10} {"p95 ms":>10}')

    # Copy matrix includes the notebook's fixed-size upper bound; computation
    # below uses --batch-size/--seq-len to avoid defaulting to giant logits.
    for shape in ((b, s), (64, 700)):
        for pinned in (False, True):
            if device.type == 'cpu' and pinned:
                continue
            host = torch.randint(0, VOCAB - 1, shape, pin_memory=pinned)
            target = torch.empty(shape, dtype=torch.long, device=device)
            measure(f'copy {shape} pinned={pinned}',
                    lambda: target.copy_(host, non_blocking=True), device, args.warmup, args.repeats)

    host = torch.randint(0, VOCAB - 1, (b, s), pin_memory=device.type != 'cpu')
    ids = host.to(device)
    model = Transformer(d, args.layers, s).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.train()
    hidden = torch.randn(b, s - 1, d, device=device)
    attn = model.blocks[0].attention
    targets = ids[:, 1:]
    logits = torch.randn(b, s - 1, VOCAB, device=device)

    with torch.no_grad():
        measure('embedding + positions', lambda: model.dropout(model.tok(ids[:, :-1]) + model.pos(
            torch.arange(s - 1, device=device))), device, args.warmup, args.repeats)
        measure('layer norm', lambda: model.norm(hidden), device, args.warmup, args.repeats)
        measure('attention (one block)', lambda: attn(hidden), device, args.warmup, args.repeats)
        measure('MLP (one block)', lambda: model.blocks[0].mlp(hidden), device, args.warmup, args.repeats)
        measure('vocab projection', lambda: model.head(hidden), device, args.warmup, args.repeats)
        measure('cross entropy', lambda: F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=PAD),
                device, args.warmup, args.repeats)
        measure('argmax + token count', lambda: ((logits.argmax(-1) == targets).sum(),
                (targets != PAD).sum()), device, args.warmup, args.repeats)

    def forward_loss(batch):
        pred = model(batch[:, :-1])
        return F.cross_entropy(pred.transpose(1, 2), batch[:, 1:], ignore_index=PAD)

    def train_step(batch):
        opt.zero_grad()
        loss = forward_loss(batch)
        loss.backward()
        opt.step()
        return loss

    measure('forward + loss', lambda: forward_loss(ids), device, args.warmup, args.repeats)
    loss = forward_loss(ids)
    measure('backward (same graph: once)', lambda: loss.backward(retain_graph=True), device,
            args.warmup, args.repeats)
    measure('Adam update', opt.step, device, args.warmup, args.repeats)
    measure('train step, preloaded', lambda: train_step(ids), device, args.warmup, args.repeats)
    measure('train step, copy in caller', lambda: train_step(host.to(device, non_blocking=True)),
            device, args.warmup, args.repeats)
    measure('loss.item() after train step', lambda: train_step(ids).item(),
            device, args.warmup, args.repeats)
    print('Note: isolated rows synchronize before/after each operation; full steps do not sync between stages.')


if __name__ == '__main__':
    main()
