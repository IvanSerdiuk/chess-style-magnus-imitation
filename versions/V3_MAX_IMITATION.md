# V3 Max Imitation (Memory-First)

## Goal

Maximize move-by-move Magnus similarity as much as possible.

## Mode

- Model: `choice_model_magnus_v2.pkl`
- Memory: enabled (exact position lookup first)
- Opening book: optional
- Model scoring used as fallback

## Run

```powershell
$sf="C:\path\to\stockfish.exe"
python evaluate_style.py --model choice_model_magnus_v2.pkl --stockfish $sf --target_pgn magnus.pgn --target_name Carlsen --k 12 --depth 10 --max_games 40 --max_plies 80 --no_book
```

## Tradeoff

- Highest imitation rate (can exceed 80% and approach ~98% on overlapping datasets).
- Larger average centipawn loss than generalized mode.
