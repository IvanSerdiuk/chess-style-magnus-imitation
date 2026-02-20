# Chess Style Imitation (Magnus-Focused)

This project trains and runs chess move-selection models that imitate a target player's move choices (currently optimized for **Carlsen**).

The system combines:
- Stockfish top-k candidate generation
- A learned move-choice model
- Optional opening book
- Optional exact position memory (max imitation mode)

## Purpose

The goal is **style imitation**, not strongest-engine play.

- If you want strongest chess: use Stockfish directly.
- If you want strongest Magnus imitation: use memory-enabled mode.

## What Is In This Repo

- `style_ranker.py`: feature extraction, compatibility helpers, memory/book lookup.
- `train_choice_model.py`: trains the v2 choice model, lambda tuning, opening-book + position-memory build.
- `evaluate_style.py`: evaluates match rate against target PGN.
- `play_full_game.py`: style side vs engine side full-game runner.
- `play_models_vs_models.py`: model vs model game runner.
- `play_vs_model.py`: human vs model.
- `demo_play.py`: single-position inspection.
- `versions/`: separated version modes and exact run recipes.

## Setup

1. Install Python 3.10+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Stockfish and note executable path.

## Data Required

This public repo ignores training data/model binaries by default (`.pgn`, `.pkl`).

Add your own files locally, e.g.:
- `magnus.pgn`
- `Nakamura.pgn`
- `others.pgn`

## Train (Magnus v2)

```powershell
$sf="C:\path\to\stockfish.exe"
python train_choice_model.py `
  --target_pgn magnus.pgn `
  --target_name Carlsen `
  --stockfish $sf `
  --out choice_model_magnus_v2.pkl `
  --k 12 --depth 10 `
  --max_games 150 --max_plies 100 `
  --model hgb `
  --book_max_plies 24 --book_min_count 2 --book_top_n 3 `
  --memory_min_count 1 --memory_top_n 1
```

## Evaluate

Max imitation mode (memory enabled):

```powershell
python evaluate_style.py --model choice_model_magnus_v2.pkl --stockfish $sf --target_pgn magnus.pgn --target_name Carlsen --k 12 --depth 10 --max_games 40 --max_plies 80 --no_book
```

Generalization-only mode (no memory, no book):

```powershell
python evaluate_style.py --model choice_model_magnus_v2.pkl --stockfish $sf --target_pgn magnus.pgn --target_name Carlsen --k 12 --depth 10 --max_games 40 --max_plies 80 --no_memory --no_book
```

## Play

Model vs model:

```powershell
python play_models_vs_models.py --white_model choice_model_magnus_v2.pkl --black_model choice_model_nakamura.pkl --stockfish $sf --k 12 --depth 10
```

Human vs model:

```powershell
python play_vs_model.py --model choice_model_magnus_v2.pkl --stockfish $sf --you white --k 12 --depth 10
```

## Version Separation

See `versions/V1_BASELINE.md`, `versions/V2_GENERALIZATION.md`, and `versions/V3_MAX_IMITATION.md`.

Those documents define:
- behavior goal
- expected tradeoffs
- exact command lines

## Important Note

Very high imitation rates (e.g. ~98%) come from exact position memory and dataset overlap.
For unseen positions, rely on no-memory evaluation to measure real generalization.
