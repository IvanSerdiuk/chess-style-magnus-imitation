# V1 Baseline (Legacy Choice Model)

## Goal

Light Magnus bias with minimal strength loss.

## Mode

- Model: `choice_model.pkl` (legacy)
- Memory: disabled
- Opening book: disabled
- Typical `lam`: `50`

## Run

```powershell
$sf="C:\path\to\stockfish.exe"
python evaluate_style.py --model choice_model.pkl --stockfish $sf --target_pgn magnus.pgn --target_name Carlsen --k 12 --depth 10 --lam 50 --window 60 --max_games 40 --max_plies 80 --no_memory --no_book
```

## Tradeoff

- Better than raw engine on some positions.
- Weaker imitation than v2/v3.
