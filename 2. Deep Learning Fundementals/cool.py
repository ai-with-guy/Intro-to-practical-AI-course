import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def new_model():
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    return model, optimizer


model, optimizer = new_model()
loss_fn = nn.BCEWithLogitsLoss()


# ------------------------------------------------------------
# Training data
# ------------------------------------------------------------

points = []   # [(x, y, class), ...]


def tensors():
    if not points:
        return None, None

    x = torch.tensor(
        [[px, py] for px, py, _ in points],
        dtype=torch.float32,
    )

    y = torch.tensor(
        [label for _, _, label in points],
        dtype=torch.float32,
    )

    return x, y


# ------------------------------------------------------------
# Decision-field grid
# ------------------------------------------------------------

RESOLUTION = 300

gx = np.linspace(-1, 1, RESOLUTION)
gy = np.linspace(-1, 1, RESOLUTION)

xx, yy = np.meshgrid(gx, gy)

grid = torch.tensor(
    np.stack([xx.ravel(), yy.ravel()], axis=1),
    dtype=torch.float32,
)


# ------------------------------------------------------------
# Matplotlib
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))


def redraw():
    ax.clear()

    # Evaluate network on every "pixel".
    with torch.no_grad():
        probability = torch.sigmoid(model(grid))
        probability = probability.reshape(RESOLUTION, RESOLUTION).numpy()

    ax.imshow(
        probability,
        extent=(-1, 1, -1, 1),
        origin="lower",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )

    # Draw the training samples.
    if points:
        p = np.array(points)

        class0 = p[:, 2] == 0
        class1 = p[:, 2] == 1

        ax.scatter(
            p[class0, 0],
            p[class0, 1],
            c="blue",
            edgecolors="white",
            s=70,
        )

        ax.scatter(
            p[class1, 0],
            p[class1, 1],
            c="red",
            edgecolors="white",
            s=70,
        )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal")

    ax.set_title(
        "Left click = blue | Right click = red | "
        "Middle click = remove | Space = train"
    )

    fig.canvas.draw_idle()


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train(steps=100):
    x, y = tensors()

    if x is None:
        return

    for _ in range(steps):
        optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

    if(loss.item()>0.0001):
        print(f"loss: {loss.item():.5f}")
    else:
        print(f"loss: {loss.item()}")



# ------------------------------------------------------------
# Matplotlib interaction
# ------------------------------------------------------------

import time
from matplotlib.widgets import Slider

plt.subplots_adjust(bottom=0.16)

slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.04])

train_steps_slider = Slider(
    ax=slider_ax,
    label="Train steps",
    valmin=1,
    valmax=100,
    valinit=10,
    valstep=1,
)


# ------------------------------------------------------------
# Mouse interaction
# ------------------------------------------------------------

def on_click(event):
    if event.inaxes != ax:
        return

    if event.xdata is None or event.ydata is None:
        return

    x = event.xdata
    y = event.ydata

    # Left click: class 0
    if event.button == 1:
        points.append((x, y, 0))

    # Right click: class 1
    elif event.button == 3:
        points.append((x, y, 1))

    # Middle click: remove nearest point
    elif event.button == 2 and points:
        distances = [
            (px - x) ** 2 + (py - y) ** 2
            for px, py, _ in points
        ]

        nearest = np.argmin(distances)

        if distances[nearest] < 0.05 ** 2:
            points.pop(nearest)

    redraw()


# ------------------------------------------------------------
# Keyboard interaction
# ------------------------------------------------------------

last_train_frame = 0.0
MIN_FRAME_TIME = 0.05


def on_key(event):
    global model, optimizer, last_train_frame

    if event.key == " ":
        now = time.monotonic()

        # Drop key-repeat events that piled up during the previous frame.
        if now - last_train_frame < MIN_FRAME_TIME:
            return

        train(int(train_steps_slider.val))
        redraw()

        last_train_frame = time.monotonic()

    elif event.key == "r":
        model, optimizer = new_model()
        redraw()

    elif event.key == "c":
        points.clear()
        redraw()


fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("key_press_event", on_key)



redraw()
plt.show()