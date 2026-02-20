#!/usr/bin/env python3
import argparse
import joblib
import chess
import chess.engine

from style_ranker import featurize_for_model, pick_book_move, pick_memory_move


def print_labeled_board(board: chess.Board):
    rows = str(board).splitlines()
    print("    a b c d e f g h")
    for i, row in enumerate(rows):
        rank = 8 - i
        print(f"{rank} | {row}")
    print("    a b c d e f g h")


def load_bundle(path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj}


def get_topk(engine, board, k=12, depth=12):
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=k)
    cand = []
    for item in info:
        mv = item["pv"][0]
        score = item["score"].pov(board.turn).score(mate_score=100000)
        cand.append((mv, score))
    cand.sort(key=lambda t: t[1], reverse=True)
    return cand


def pick_style_move(
    bundle,
    model,
    engine,
    board,
    k,
    depth,
    lam,
    window_cp,
    use_memory,
    use_book,
    book_window_cp,
):
    cand = get_topk(engine, board, k=k, depth=depth)
    best_mv, best_sc = cand[0]

    if use_memory:
        mem_mv, mem_meta = pick_memory_move(bundle, board)
        if mem_mv is not None:
            mem_sc = next((sc for mv, sc in cand if mv == mem_mv), best_sc)
            return {
                "move": mem_mv,
                "eng_sc": mem_sc,
                "p": None,
                "val": None,
                "sf_mv": best_mv,
                "sf_sc": best_sc,
                "cand": cand,
                "source": "memory",
                "memory": mem_meta,
                "book": None,
            }

    if use_book:
        book_mv, book_meta = pick_book_move(bundle, board, cand, max_loss_cp=book_window_cp)
        if book_mv is not None:
            book_sc = next(sc for mv, sc in cand if mv == book_mv)
            return {
                "move": book_mv,
                "eng_sc": book_sc,
                "p": None,
                "val": None,
                "sf_mv": best_mv,
                "sf_sc": best_sc,
                "cand": cand,
                "source": "book",
                "memory": None,
                "book": book_meta,
            }

    cand2 = [t for t in cand if best_sc - t[1] <= window_cp]
    if len(cand2) >= 2:
        cand = cand2

    best = None
    best_val = float("-inf")
    n_cand = len(cand)
    for rank, (mv, eng_sc) in enumerate(cand):
        x = featurize_for_model(model, board, mv, eng_sc, best_sc, rank, n_cand)
        p = float(model.predict_proba([x])[0, 1])
        val = eng_sc + lam * p
        if val > best_val:
            best_val = val
            best = {
                "move": mv,
                "eng_sc": eng_sc,
                "p": p,
                "val": val,
                "sf_mv": best_mv,
                "sf_sc": best_sc,
                "cand": cand,
                "source": "model",
                "memory": None,
                "book": None,
            }

    return best


def parse_user_move(board, s):
    s = s.strip()
    if s.lower() in {"quit", "exit"}:
        return None

    try:
        mv = chess.Move.from_uci(s)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass

    try:
        mv = board.parse_san(s)
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass

    return "invalid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Your trained model .pkl")
    ap.add_argument("--stockfish", required=True, help="Path to stockfish exe")
    ap.add_argument("--you", choices=["white", "black"], default="white")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--book_window", type=int, default=80)
    ap.add_argument("--book_max_plies", type=int, default=24)
    ap.add_argument("--no_memory", action="store_true")
    ap.add_argument("--no_book", action="store_true")
    ap.add_argument("--start_fen", default=None, help="Optional starting FEN")
    ap.add_argument("--uci", action="store_true", help="Display moves in UCI only")
    args = ap.parse_args()

    bundle = load_bundle(args.model)
    model = bundle["model"]
    lam = float(args.lam) if args.lam is not None else float(bundle.get("recommended_lam", 120.0))
    bundle_book_max = int(bundle.get("opening_book_max_plies", args.book_max_plies))

    board = chess.Board() if args.start_fen is None else chess.Board(args.start_fen)
    you_are_white = (args.you == "white")

    print("\nType moves as SAN (e.g., Nf3, exd5, O-O) or UCI (e.g., e2e4).")
    print("Type 'quit' to stop.\n")
    print(
        f"Model config: lam={lam:g}, window={args.window}cp, "
        f"book_window={args.book_window}cp, use_memory={not args.no_memory}, "
        f"use_book={not args.no_book}"
    )

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    ply = 0
    try:
        while not board.is_game_over():
            print_labeled_board(board)
            print("FEN:", board.fen())
            print("Turn:", "White" if board.turn == chess.WHITE else "Black")
            print("-" * 60)

            human_turn = (board.turn == chess.WHITE and you_are_white) or (
                board.turn == chess.BLACK and not you_are_white
            )

            if human_turn:
                s = input("Your move: ")
                mv = parse_user_move(board, s)
                if mv is None:
                    print("Bye.")
                    return
                if mv == "invalid":
                    print("Invalid move. Try again.\n")
                    continue

                board.push(mv)
                ply += 1
                print(f"You played: {mv.uci() if args.uci else board.peek()}\n")
            else:
                allow_memory = not args.no_memory
                allow_book = (not args.no_book) and (ply < min(args.book_max_plies, bundle_book_max))
                pick = pick_style_move(
                    bundle,
                    model,
                    engine,
                    board,
                    args.k,
                    args.depth,
                    lam,
                    args.window,
                    allow_memory,
                    allow_book,
                    args.book_window,
                )

                mv = pick["move"]
                print("=== Engine choice ===")
                print(f"Stockfish best: {pick['sf_mv'].uci()} (score {pick['sf_sc']})")

                if pick["source"] == "memory":
                    m = pick["memory"]
                    print(
                        f"Model plays:    {mv.uci()} (MEM freq={m['freq']:.2f}, "
                        f"count={m['count']}, total={m['total']})"
                    )
                elif pick["source"] == "book":
                    b = pick["book"]
                    print(
                        f"Model plays:    {mv.uci()} (BOOK freq={b['freq']:.2f}, "
                        f"count={b['count']}, gap={b['gap_cp']:.0f}cp)"
                    )
                else:
                    print(
                        f"Model plays:    {mv.uci()} "
                        f"(engine {pick['eng_sc']}, styleP {pick['p']:.3f}, combined {pick['val']:.2f})"
                    )

                print("\nCandidates used:", len(pick["cand"]), f"(window={args.window}cp)")
                print("Top candidates:")
                n_cand = len(pick["cand"])
                for rank, (cmv, csc) in enumerate(pick["cand"][:10]):
                    x = featurize_for_model(model, board, cmv, csc, pick["sf_sc"], rank, n_cand)
                    cp = float(model.predict_proba([x])[0, 1])
                    cval = csc + lam * cp
                    print(f"  {cmv.uci():8s}  eng={csc:6d}  p={cp:.3f}  comb={cval:.2f}")
                print()

                board.push(mv)
                ply += 1
                print(f"Model played: {mv.uci() if args.uci else board.peek()}\n")

        print(board)
        print("\n=== Game over ===")
        print("Result:", board.result())
        print("Termination:", board.outcome())

    finally:
        engine.quit()


if __name__ == "__main__":
    main()
