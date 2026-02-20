#!/usr/bin/env python3
import argparse
import joblib
import chess
import chess.engine

from style_ranker import featurize_for_model


def get_topk(engine, board, k=8, depth=12):
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=k)
    cand = []
    for item in info:
        mv = item["pv"][0]
        score = item["score"].pov(board.turn).score(mate_score=100000)
        cand.append((mv, score))
    cand.sort(key=lambda t: t[1], reverse=True)
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--fen", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--window", type=int, default=30,
                    help="Keep only engine moves within this many centipawns of best.")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    lam = float(args.lam) if args.lam is not None else float(bundle.get("recommended_lam", 120.0) if isinstance(bundle, dict) else 120.0)

    board = chess.Board() if args.fen is None else chess.Board(args.fen)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        cand = get_topk(engine, board, args.k, args.depth)
        best_engine_move, best_engine_score = cand[0]

        cand_filtered = [t for t in cand if best_engine_score - t[1] <= args.window]
        if len(cand_filtered) >= 2:
            cand = cand_filtered

        best = None
        best_val = float("-inf")
        rows = []
        n_cand = len(cand)

        for rank, (mv, eng_score) in enumerate(cand):
            x = featurize_for_model(model, board, mv, eng_score, best_engine_score, rank, n_cand)
            style_p = float(model.predict_proba([x])[0, 1])
            combined = eng_score + lam * style_p
            rows.append((mv.uci(), eng_score, style_p, combined))
            if combined > best_val:
                best_val = combined
                best = (mv, eng_score, style_p, combined)

        style_mv, style_eng, style_p, combo = best

        print("\n=== Position ===")
        print(board)
        print("\nFEN:", board.fen())

        print("\n=== Decision ===")
        print(f"Stockfish best: {best_engine_move.uci()} (score {best_engine_score})")
        print(f"Style-biased:   {style_mv.uci()} (score {style_eng}, styleP {style_p:.3f}, combined {combo:.2f})")

        print(f"\nConfig: lam={lam:g}, window={args.window} cp")
        print(f"Candidates used: {len(cand)}")
        print("\nTop candidates (uci | engine_score | style_prob | combined):")
        for uci, es, sp, cb in rows:
            print(f"{uci:8s}  {es:8d}   {sp:.3f}     {cb:.2f}")

    finally:
        engine.quit()


if __name__ == "__main__":
    main()
