# Intro to Practical AI Course 🤖📊

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/package--manager-uv-7f52ff.svg)](https://github.com/astral-sh/uv)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-orange.svg)](https://jupyter.org/)

Hands-on practical Artificial Intelligence and Data Science course repository powered by `uv` and Jupyter.

---

## ⚡ Quick Start

### 1. Prerequisites
Install `uv` (fast Python package manager):

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Launch the Workshop Notebook

#### Linux / macOS:
```bash
./launch.sh
```

#### Windows:
```cmd
Launch.bat
```

#### Manual Start:
```bash
uv run jupyter lab "Making sense of Big Data/Workshop1.ipynb"
```

Running the launcher script automatically boots up Jupyter Lab with `Workshop1.ipynb` open and ready to run.

---

## 📂 Project Structure

```text
Intro-to-practical-AI-course/
├── launch.sh                     # Linux/macOS one-click launcher
├── Launch.bat                    # Windows one-click launcher
├── pyproject.toml                # Project configuration and dependencies
├── uv.lock                       # Lockfile managed by uv
├── src/                          # Course materials and reference notebooks
├── Making sense of Big Data/     # Workshop directory
│   ├── Workshop1.ipynb           # Main interactive workshop notebook
│   ├── utils.py                  # Exercise validation helper functions
│   └── titanic.csv               # Practical exercise dataset
└── README.md                     # Project documentation
```

---

## 💻 Running in VS Code / IDE

If you prefer executing notebooks directly inside VS Code or Antigravity IDE:

1. Open this repository folder (`Intro-to-practical-AI-course`) in your IDE.
2. Open `Making sense of Big Data/Workshop1.ipynb`.
3. Select the Jupyter Kernel in the top-right corner:
   - Select **Python Environments...** $\rightarrow$ `.venv/bin/python` (or `Python 3 (.venv)`).
4. Run cells interactively!

---

## 📦 Environment & Dependencies

Dependencies are managed at the root level via `pyproject.toml` using `uv`:
- **`jupyter`**: Interactive notebook environment
- **`pandas`**: Data analysis and manipulation
- **`numpy`**: High-performance numerical computing
- **`matplotlib`**: Data visualization
- **`scikit-learn`**: Machine learning utilities
- **`torch` & `torchvision`**: Deep learning framework
