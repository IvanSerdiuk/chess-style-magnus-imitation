#!/usr/bin/env python3
import argparse

import joblib

import chess
import chess.pgn
import chess.engine

import numpy as np

from style_ranker import featurize_for_model, pick_book_move, pick_memory_move


def get_topk(engine, board, k=8, depth=10):
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=k)
    cand = []
    for item in info:
        mv = item["pv"][0]
        score = item["score"].pov(board.turn).score(mate_score=100000)
        cand.append((mv, score))
    cand.sort(key=lambda t: t[1], reverse=True)
    return cand


def choose_style(bundle, model, board, cand, lam, window_cp, use_memory, use_book, book_window_cp):
    best_engine_score = cand[0][1]
    cand_used = cand

    if use_memory:
        mem_mv, _mem_meta = pick_memory_move(bundle, board)
        if mem_mv is not None:
            mem_sc = next((sc for mv, sc in cand if mv == mem_mv), best_engine_score)
            return (mem_mv, mem_sc, 1.0, mem_sc + lam), cand

    if use_book:
        book_mv, _book_meta = pick_book_move(bundle, board, cand, max_loss_cp=book_window_cp)
        if book_mv is not None:
            book_sc = next(sc for mv, sc in cand if mv == book_mv)
            return (book_mv, book_sc, 1.0, book_sc + lam), cand

    if window_cp is not None and window_cp >= 0:
        filtered = [t for t in cand if best_engine_score - t[1] <= window_cp]
        if len(filtered) >= 2:
            cand_used = filtered

    best = None
    best_val = -1e18
    n_cand = len(cand_used)
    for rank, (mv, eng_score) in enumerate(cand_used):
        x = featurize_for_model(model, board, mv, eng_score, best_engine_score, rank, n_cand)
        style_p = float(model.predict_proba([x])[0, 1])
        combined = eng_score + lam * style_p
        if combined > best_val:
            best_val = combined
            best = (mv, eng_score, style_p, combined)
    return best, cand_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--target_pgn", required=True)
    ap.add_argument("--target_name", default="Carlsen")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--book_window", type=int, default=80)
    ap.add_argument("--book_max_plies", type=int, default=24)
    ap.add_argument("--no_memory", action="store_true")
    ap.add_argument("--no_book", action="store_true")
    ap.add_argument("--max_games", type=int, default=60)
    ap.add_argument("--max_plies", type=int, default=80)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    lam = float(args.lam) if args.lam is not None else float(bundle.get("recommended_lam", 120.0) if isinstance(bundle, dict) else 120.0)
    bundle_book_max = int(bundle.get("opening_book_max_plies", args.book_max_plies) if isinstance(bundle, dict) else args.book_max_plies)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    total = 0
    topk_cover = 0
    match_engine = 0
    match_style = 0
    eval_drop = []

    try:
        with open(args.target_pgn, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(args.max_games):
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                white = game.headers.get("White", "")
                black = game.headers.get("Black", "")

                board = game.board()
                ply = 0
                for true_mv in game.mainline_moves():
                    if ply >= args.max_plies:
                        break
                    if true_mv not in board.legal_moves:
                        break

                    is_target_turn = (
                        (board.turn == chess.WHITE and args.target_name in white)
                        or (board.turn == chess.BLACK and args.target_name in black)
                    )

                    if not is_target_turn:
                        board.push(true_mv)
                        ply += 1
                        continue

                    cand = get_topk(engine, board, args.k, args.depth)
                    cand_moves = [mv for mv, _ in cand]
                    engine_mv, engine_score = cand[0]
                    allow_memory = not args.no_memory
                    allow_book = (not args.no_book) and (ply < min(args.book_max_plies, bundle_book_max))
                    (style_mv, style_score, _p, _combined), _cand_used = choose_style(
                        bundle, model, board, cand, lam, args.window, allow_memory, allow_book, args.book_window
                    )

                    total += 1

                    if true_mv in cand_moves:
                        topk_cover += 1

                    if engine_mv == true_mv:
                        match_engine += 1

                    if style_mv == true_mv:
                        match_style += 1

                    eval_drop.append(engine_score - style_score)

                    board.push(true_mv)
                    ply += 1

        print("\n=== Evaluation (positions where target to move) ===")
        print(f"Model: {args.model}")
        print(f"Target: {args.target_name}")
        print(
            f"Config: k={args.k} depth={args.depth} lam={lam:g} window={args.window}cp "
            f"book_window={args.book_window}cp use_memory={not args.no_memory} "
            f"use_book={not args.no_book}"
        )
        print(f"Positions tested: {total}")
        print(f"Top-k coverage: {topk_cover/max(total, 1):.3f}")
        print(f"Stockfish top-1 match rate: {match_engine/max(total, 1):.3f}")
        print(f"Style-biased match rate:    {match_style/max(total, 1):.3f}")
        print(f"Avg strength loss (cp):     {float(np.mean(eval_drop)) if eval_drop else 0.0:.2f}")
        print(f"Median strength loss (cp):  {float(np.median(eval_drop)) if eval_drop else 0.0:.2f}")

    finally:
        engine.quit()


if __name__ == "__main__":
    main()
