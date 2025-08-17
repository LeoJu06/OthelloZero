# OthelloZero

**OthelloZero** is the practical part of my baccalaureate thesis. The goal is to build an **AlphaZero-style** system that learns the board game [Othello (Reversi)](https://de.wikipedia.org/wiki/Othello_(Spiel)) via **self-play** and **deep reinforcement learning**.  
The system combines a **Policy+Value neural network** with **Monte-Carlo Tree Search (MCTS)** to discover strategies autonomously and improve iteratively.

---

## 🚀 Quick Start

> **Requirements**
> - Python 3.9+  
> - (Recommended) a virtual environment  
> - Dependencies from `requirements.txt`

### 1) Clone & Install
```bash
git clone https://github.com/<LeoJu06>/OthelloZero.git
cd OthelloZero

# (optional) create & activate a virtual env
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2) Start Training
Training is launched via the **pipeline module**:

```bash
# Preferred (module form, run from the project root):
python -m main.pipeline

# Alternative (if you prefer direct script execution):
python main/pipeline.py
```

> **Tip:** Always run from the **project root** (the folder containing `main/` and `src/`).

### 3) Play Human vs AI (against your trained model)
```bash
# Ensure 'src' is on the Python path:
# Linux/macOS:
export PYTHONPATH=src:$PYTHONPATH
# Windows (PowerShell):
$env:PYTHONPATH="src;$env:PYTHONPATH"

# Launch the game:
python src/othello/gamePvsAi.py
```

If the script supports options (e.g., model checkpoint path, number of MCTS simulations, UI speed), check the header of `gamePvsAi.py` or its argument parser for available flags.

---

## 🧠 What’s Inside

- **Self-Play:** continuously generates fresh training positions.  
- **MCTS:** explores action trajectories and guides move selection.  
- **Policy+Value Network:** evaluates states and suggests move probabilities.  
- **Iterative Loop:** _Self-Play → Training → Evaluation → (optionally) Promote new model_.

---

## 🗂️ Project Structure (excerpt)

```text
OthelloZero/
├─ main/
│  └─ pipeline.py          # Entry point for training (or run as module: main.pipeline)
├─ src/
│  └─ othello/
│     ├─ game.py
│     ├─ gamePvsAi.py      # Human vs AI
│     └─ ...
├─ requirements.txt
└─ README.md
```

> Some scripts expect `src` in `PYTHONPATH`. See the **Play** section above.

---

## ⚙️ Configuration (common places to look)
- **Training parameters:** usually near the top of `main/pipeline.py` or in a config block/arg parser there.  
- **Model/checkpoint paths:** defined in the pipeline or passed as CLI args.  
- **MCTS settings (e.g., simulations, exploration constants):** inside MCTS module or pipeline args.

---

## 🧪 Troubleshooting

- **`ModuleNotFoundError` for project modules**  
  Run commands from the project root and ensure `PYTHONPATH` includes `src`.

- **Import/version conflicts**  
  Recreate a clean virtual environment and reinstall `requirements.txt`.

- **Checkpoint not found / paths**  
  Verify your training produced checkpoints and that your `gamePvsAi.py` points to the correct file.

- **Slow training or high memory usage**  
  Reduce MCTS simulations, batch size, or model size in the training config.

---

## 🙏 Acknowledgments

- Inspired by the **AlphaZero** paradigm (Policy/Value networks guided by MCTS).  
- Othello (Reversi) rules and community resources.

---
