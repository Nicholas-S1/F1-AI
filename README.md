# F1 AI

CPSC 460/560 final project — reinforcement learning F1 driver built with pygame.

## Setup (Windows)

Requires [Python 3.10+](https://www.python.org/downloads/) (tick "Add python.exe to PATH") and [Git](https://git-scm.com/download/win).

**First time (cloning):**
```
git clone <repo-url>
cd F1-AI
setup.bat
```

`setup.bat` creates a `.venv` virtual environment, installs everything in `requirements.txt` (pygame, numpy, matplotlib), and points VS Code at the venv.

Then open the folder in VS Code (`code .`) and install the **Python** extension if prompted. New terminals in VS Code will auto-activate `.venv`.

**Manual setup instead of setup.bat:**
```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Setup (macOS)

pygame 2.6 has no prebuilt wheel for Python 3.14 yet, so use Python 3.13 (or 3.10–3.12). With [Homebrew](https://brew.sh):
```
brew install python@3.13
git clone <repo-url>
cd F1-AI
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

With the venv activated, from the project folder:
```
python main.py
```
On Windows, `.venv\Scripts\activate` first; on macOS, `source .venv/bin/activate`.

Adding a package: `pip install <pkg>`, then add it to `requirements.txt` so your partner gets it too.

## Team & responsibilities

| Member | Responsibilities |
|---|---|
| Nick Setters | |
| | |
