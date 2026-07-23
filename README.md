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
launch.bat
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
├── launch.sh                     # Linux/macOS launcher
├── launch.bat                    # Windows launcher
├── src/                          # Course materials and reference notebooks
└── Making sense of Big Data/     # Workshop directory
    ├── Workshop1.ipynb           # Main interactive workshop notebook
    ├── utils.py                  # Exercise validation helper functions
    └── titanic.csv               # Practical exercise dataset
```
