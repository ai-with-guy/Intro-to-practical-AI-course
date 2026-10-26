import unittest

import torch
from torch import nn

from training_history import TrainingHistory


class TrainingHistoryTests(unittest.TestCase):
    def check_timeline(self, device):
        model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)).to(device)
        inputs = torch.randn(8, 2, device=device)
        optimizer = torch.optim.Adam(model.parameters())
        history = TrainingHistory(model)
        initial = model(inputs).detach().clone()
        for _ in range(3):
            optimizer.zero_grad()
            model(inputs).square().mean().backward()
            optimizer.step()
            history.record(model)
        final = model(inputs).detach().clone()
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        steps = [state['step'].clone() for state in optimizer.state.values()]

        self.assertEqual(history.last_epoch, 3)
        torch.testing.assert_close(history.predict(model, inputs.cpu(), 0), initial)
        torch.testing.assert_close(history.predict(model, inputs.cpu(), 3), final)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[key])
        for step, state in zip(steps, optimizer.state.values()):
            torch.testing.assert_close(step, state['step'])
        for state in history.states:
            self.assertTrue(all(value.device == inputs.device for value in state.values()))
            self.assertTrue(all(not value.requires_grad for value in state.values()))

    def test_cpu(self):
        self.check_timeline('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self.check_timeline('cuda')


if __name__ == '__main__':
    unittest.main()
