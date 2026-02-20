# V2 Generalization (Learned Model Only)

## Goal

Use the stronger v2 model while measuring genuine generalization.

## Mode

- Model: `choice_model_magnus_v2.pkl`
- Memory: disabled
- Opening book: disabled
- `lam`: auto from model bundle (typically `250`) or override manually

## Run

```powershell
$sf="C:\path\to\stockfish.exe"
python evaluate_style.py --model choice_model_magnus_v2.pkl --stockfish $sf --target_pgn magnus.pgn --target_name Carlsen --k 12 --depth 10 --max_games 40 --max_plies 80 --no_memory --no_book
```

## Tradeoff

- Real learned behavior.
- Lower imitation rate than memory mode, but stronger out-of-sample meaning.
