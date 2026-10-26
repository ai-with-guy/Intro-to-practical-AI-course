"""Small, device-local weight snapshots for read-only training timelines."""

import torch


class TrainingHistory:
    def __init__(self, model):
        self.states = []
        self.record(model)  # Epoch zero is the untrained model.

    def record(self, model):
        self.states.append({
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        })

    @property
    def last_epoch(self):
        return len(self.states) - 1

    @torch.no_grad()
    def predict(self, model, inputs, epoch):
        # functional_call uses the snapshot without changing live weights or Adam.
        state = self.states[int(epoch)]
        device = next(iter(state.values())).device
        return torch.func.functional_call(model, state, (inputs.to(device),))
