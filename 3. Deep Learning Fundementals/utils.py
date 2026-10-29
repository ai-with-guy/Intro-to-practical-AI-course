import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import gradio as gr

def display_points(data, title, ax=None):
    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(7, 5))

    for label, colour in [(0, "tab:blue"), (1, "tab:orange")]:
        rows = data[data["class"] == label]
        ax.scatter(rows["x1"], rows["x2"], c=colour, label=f"class {label}", edgecolors="white")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.2)

    if own_figure:
        plt.show()


def display_classifications(model, data, title):
    x1 = np.linspace(data["x1"].min() - 0.5, data["x1"].max() + 0.5, 250)
    x2 = np.linspace(data["x2"].min() - 0.5, data["x2"].max() + 0.5, 250)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    grid = torch.tensor(np.column_stack((grid_x1.ravel(), grid_x2.ravel())), dtype=torch.float32)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(grid)).numpy().reshape(grid_x1.shape)

    _, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(grid_x1, grid_x2, probabilities, levels=np.linspace(0, 1, 11), cmap="coolwarm", alpha=0.35)
    ax.contour(grid_x1, grid_x2, probabilities, levels=[0.5], colors="black", linewidths=2)
    display_points(data, title, ax=ax)
    plt.show()


def display_classifications_history(model, data, title):
    x1 = np.linspace(data["x1"].min() - 0.5, data["x1"].max() + 0.5, 250)
    x2 = np.linspace(data["x2"].min() - 0.5, data["x2"].max() + 0.5, 250)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    grid = torch.tensor(np.column_stack((grid_x1.ravel(), grid_x2.ravel())), dtype=torch.float32)

    history = model.training_history
    features = torch.tensor(data[["x1", "x2"]].values, dtype=torch.float32)
    targets = torch.tensor(data["class"].values, dtype=torch.float32).reshape(-1, 1)

    def show_epoch(epoch):
        epoch = int(epoch)
        probabilities = history.predict(model, grid, epoch).sigmoid().cpu().numpy().reshape(grid_x1.shape)
        logits = history.predict(model, features, epoch).cpu()
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets).item()
        accuracy = ((logits >= 0) == targets.bool()).float().mean().item()
        figure, ax = plt.subplots(figsize=(7, 5))
        ax.contourf(grid_x1, grid_x2, probabilities, levels=np.linspace(0, 1, 11), cmap="coolwarm", alpha=0.35)
        if probabilities.min() < 0.5 < probabilities.max():
            ax.contour(grid_x1, grid_x2, probabilities, levels=[0.5], colors="black", linewidths=2)
        display_points(data, f"{title} — epoch {epoch}/{history.last_epoch}", ax=ax)
        plt.close(figure)
        return figure, f"Loss: {loss:.5f} | Accuracy: {accuracy:.1%}"

    with gr.Blocks(title=title) as timeline:
        figure, metrics = show_epoch(0)
        status = gr.Markdown(metrics)
        epoch = gr.Slider(0, max(1, history.last_epoch), value=1, step=1, label="Epoch", interactive=history.last_epoch > 0)
        plot = gr.Plot(figure)
        epoch.change(show_epoch, inputs=epoch, outputs=[plot, status], trigger_mode="always_last")
        
    # Leave room for the full plot and controls inside the notebook iframe.
    timeline.launch(quiet=True, prevent_thread_lock=True, height=850)
    #return timeline


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