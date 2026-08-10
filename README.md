# Intro to Practical AI Course 🤖📊

[![Python](https://img.shields.io/badge/Python-3.14%2B-blue.svg)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/package--manager-uv-7f52ff.svg)](https://github.com/astral-sh/uv)
[![Jupyter](https://img.shields.io/badge/Jupyter-Lab-orange.svg)](https://jupyter.org/)
[Env Manager: micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)


This workshop covers many practical concepts for ML software engineering in the real world.
It is intended for advance students who have already mastered python.
For all leasons but leason 0 we assume basic competency with virtual enviorments. 

---

### 1. Prerequisites
We will be using `uv` as our prefered package manager. 
If you are competent with something else thats fine. But some of the util scripts wont work.

To install `uv` you can:

**Linux / macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

or if you are using conda/micromamba you can simply
```bash
micromamba install uv
```

and it will be added to your venv similar to how pip works


