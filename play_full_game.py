#!/usr/bin/env python3
import argparse
import joblib
import chess
import chess.engine

from style_ranker import featurize_for_model, pick_book_move, pick_memory_move


def get_topk(engine, board, k=8, depth=12):
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=k)
    cand = []
    for item in info:
        mv = item["pv"][0]
        score = item["score"].pov(board.turn).score(mate_score=100000)
        cand.append((mv, score))
    cand.sort(key=lambda t: t[1], reverse=True)
    return cand


def choose_style_move(
    bundle,
    model,
    board,
    cand,
    lam,
    window_cp,
    use_memory,
    use_book,
    book_window_cp,
):
    best_engine_score = cand[0][1]

    if use_memory:
        mem_mv, mem_meta = pick_memory_move(bundle, board)
        if mem_mv is not None:
            mem_sc = next((sc for mv, sc in cand if mv == mem_mv), best_engine_score)
            return mem_mv, {
                "source": "memory",
                "eng_score": mem_sc,
                "style_p": None,
                "combined": None,
                "rank": None,
                "memory": mem_meta,
                "book": None,
            }

    if use_book:
        book_mv, book_meta = pick_book_move(bundle, board, cand, max_loss_cp=book_window_cp)
        if book_mv is not None:
            book_sc = next(sc for mv, sc in cand if mv == book_mv)
            return book_mv, {
                "source": "book",
                "eng_score": book_sc,
                "style_p": None,
                "combined": None,
                "rank": None,
                "memory": None,
                "book": book_meta,
            }

    if window_cp is not None and window_cp >= 0:
        cand_filtered = [t for t in cand if best_engine_score - t[1] <= window_cp]
        if len(cand_filtered) >= 2:
            cand = cand_filtered

    best_mv = None
    best_val = -1e18
    best_details = None
    n_cand = len(cand)

    for rank, (mv, eng_score) in enumerate(cand):
        x = featurize_for_model(model, board, mv, eng_score, best_engine_score, rank, n_cand)
        style_p = float(model.predict_proba([x])[0, 1])
        combined = eng_score + lam * style_p
        if combined > best_val:
            best_val = combined
            best_mv = mv
            best_details = {
                "source": "model",
                "eng_score": eng_score,
                "style_p": style_p,
                "combined": combined,
                "rank": rank,
                "memory": None,
                "book": None,
            }

    return best_mv, best_details


def should_engine_play(side, style_side, engine_side):
    if engine_side == "both":
        return True
    if side == engine_side:
        return True
    if side == style_side:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--book_window", type=int, default=80)
    ap.add_argument("--book_max_plies", type=int, default=24)
    ap.add_argument("--no_memory", action="store_true")
    ap.add_argument("--no_book", action="store_true")
    ap.add_argument("--max_plies", type=int, default=120)
    ap.add_argument("--style_side", choices=["white", "black"], default="white")
    ap.add_argument("--engine_side", choices=["white", "black", "both"], default="black")
    ap.add_argument("--print_every", type=int, default=1)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle

    lam = (
        float(args.lam)
        if args.lam is not None
        else float(bundle.get("recommended_lam", 120.0) if isinstance(bundle, dict) else 120.0)
    )
    bundle_book_max = int(bundle.get("opening_book_max_plies", args.book_max_plies) if isinstance(bundle, dict) else args.book_max_plies)

    print("Config:")
    print(f"  lam={lam:g}  window={args.window}cp  style_side={args.style_side}  engine_side={args.engine_side}")
    print(
        f"  book_window={args.book_window}cp  book_max_plies={args.book_max_plies} "
        f"(bundle max {bundle_book_max})  use_memory={not args.no_memory} "
        f"use_book={not args.no_book}"
    )

    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    try:
        ply = 0
        while not board.is_game_over() and ply < args.max_plies:
            side = "white" if board.turn == chess.WHITE else "black"

            if should_engine_play(side, args.style_side, args.engine_side):
                mv = engine.play(board, chess.engine.Limit(depth=args.depth)).move
                tag = f"ENGINE({side})"
                extra = f"depth={args.depth}"
            else:
                cand = get_topk(engine, board, args.k, args.depth)
                allow_memory = not args.no_memory
                allow_book = (not args.no_book) and (ply < min(args.book_max_plies, bundle_book_max))
                mv, details = choose_style_move(
                    bundle,
                    model,
                    board,
                    cand,
                    lam,
                    args.window,
                    allow_memory,
                    allow_book,
                    args.book_window,
                )
                tag = f"STYLE({side})"

                if details["source"] == "memory":
                    m = details["memory"]
                    extra = (
                        f"MEM freq={m['freq']:.2f}, count={m['count']}, total={m['total']}"
                    )
                elif details["source"] == "book":
                    b = details["book"]
                    extra = (
                        f"BOOK freq={b['freq']:.2f}, count={b['count']}, "
                        f"gap={b['gap_cp']:.0f}cp"
                    )
                else:
                    extra = (
                        f"eng={details['eng_score']}, styleP={details['style_p']:.3f}, "
                        f"comb={details['combined']:.2f}, lam={lam:g}, rank={details['rank']}"
                    )

            board.push(mv)
            ply += 1

            if ply % args.print_every == 0:
                print(f"{ply:03d}. {tag}: {mv.uci()}  [{extra}]")
                print(board)
                print("FEN:", board.fen())
                print("-" * 60)

        print("\n=== Game over ===")
        print("Result:", board.result(claim_draw=True))
        print("Outcome:", board.outcome(claim_draw=True))
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
