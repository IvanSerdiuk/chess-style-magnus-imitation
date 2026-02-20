import argparse
import joblib
import chess
import chess.engine

from style_ranker import featurize_for_model, pick_book_move, pick_memory_move


def load_bundle(path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        return obj
    return {"model": obj}


def analyse_topk(engine, board, k, depth):
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=k)
    cand = []
    for item in info:
        mv = item["pv"][0]
        sc = item["score"].pov(board.turn).score(mate_score=100000)
        cand.append((mv, sc))
    cand.sort(key=lambda t: t[1], reverse=True)
    return cand


def pick_move(
    board,
    engine,
    bundle,
    model,
    k,
    depth,
    lam,
    window_cp,
    use_memory,
    use_book,
    book_window_cp,
):
    cand = analyse_topk(engine, board, k, depth)
    best_sc = cand[0][1]

    if use_memory:
        mem_mv, mem_meta = pick_memory_move(bundle, board)
        if mem_mv is not None:
            mem_sc = next((sc for mv, sc in cand if mv == mem_mv), best_sc)
            return {
                "move": mem_mv,
                "engine_score": mem_sc,
                "style_prob": None,
                "combined": None,
                "rank": None,
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
                "engine_score": book_sc,
                "style_prob": None,
                "combined": None,
                "rank": None,
                "source": "book",
                "memory": None,
                "book": book_meta,
            }

    if window_cp is not None and window_cp >= 0:
        windowed = [t for t in cand if best_sc - t[1] <= window_cp]
        if len(windowed) >= 2:
            cand = windowed

    scored = []
    n_cand = len(cand)
    for rank, (mv, sc) in enumerate(cand):
        x = featurize_for_model(model, board, mv, sc, best_sc, rank, n_cand)
        p = float(model.predict_proba([x])[0, 1])
        combined = sc + lam * p
        scored.append((combined, mv, sc, p, rank))

    scored.sort(key=lambda t: t[0], reverse=True)
    best = scored[0]
    return {
        "move": best[1],
        "engine_score": best[2],
        "style_prob": best[3],
        "combined": best[0],
        "rank": best[4],
        "source": "model",
        "memory": None,
        "book": None,
    }


def side_name(turn):
    return "WHITE" if turn == chess.WHITE else "BLACK"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--white_model", required=True)
    ap.add_argument("--black_model", required=True)
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--lam", type=float, default=120)
    ap.add_argument("--white_lam", type=float, default=None)
    ap.add_argument("--black_lam", type=float, default=None)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--book_window", type=int, default=80)
    ap.add_argument("--book_max_plies", type=int, default=24)
    ap.add_argument("--no_memory", action="store_true")
    ap.add_argument("--no_book", action="store_true")
    ap.add_argument("--max_plies", type=int, default=200)
    ap.add_argument("--print_every", type=int, default=2)
    args = ap.parse_args()

    white_bundle = load_bundle(args.white_model)
    black_bundle = load_bundle(args.black_model)

    white_model = white_bundle["model"]
    black_model = black_bundle["model"]

    white_lam = (
        args.white_lam
        if args.white_lam is not None
        else float(white_bundle.get("recommended_lam", args.lam))
    )
    black_lam = (
        args.black_lam
        if args.black_lam is not None
        else float(black_bundle.get("recommended_lam", args.lam))
    )

    print("Config:")
    print(f"  white_lam={white_lam:g}  black_lam={black_lam:g}")
    print(
        f"  window={args.window}cp  book_window={args.book_window}cp  "
        f"book_max_plies={args.book_max_plies}  use_memory={not args.no_memory} "
        f"use_book={not args.no_book}"
    )

    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    ply = 0
    try:
        while not board.is_game_over() and ply < args.max_plies:
            if board.turn == chess.WHITE:
                bundle = white_bundle
                model = white_model
                lam = white_lam
            else:
                bundle = black_bundle
                model = black_model
                lam = black_lam

            bundle_book_max = int(bundle.get("opening_book_max_plies", args.book_max_plies))
            allow_memory = not args.no_memory
            allow_book = (not args.no_book) and (ply < min(args.book_max_plies, bundle_book_max))

            pick = pick_move(
                board,
                engine,
                bundle,
                model,
                args.k,
                args.depth,
                lam,
                args.window,
                allow_memory,
                allow_book,
                args.book_window,
            )
            mv = pick["move"]

            if ply % args.print_every == 0:
                side = side_name(board.turn)
                if pick["source"] == "memory":
                    m = pick["memory"]
                    print(
                        f"{ply+1:3d}. {side}: {mv.uci()}  "
                        f"(MEM freq={m['freq']:.2f}, count={m['count']}, total={m['total']})"
                    )
                elif pick["source"] == "book":
                    b = pick["book"]
                    print(
                        f"{ply+1:3d}. {side}: {mv.uci()}  "
                        f"(BOOK freq={b['freq']:.2f}, count={b['count']}, gap={b['gap_cp']:.0f}cp)"
                    )
                else:
                    print(
                        f"{ply+1:3d}. {side}: {mv.uci()}  "
                        f"(engine {pick['engine_score']}, styleP {pick['style_prob']:.3f}, "
                        f"lam {lam:g}, rank {pick['rank']})"
                    )
                print(board, "\n")

            board.push(mv)
            ply += 1

        print("\n=== Game over ===")
        print("Result:", board.result())
        print("Outcome:", board.outcome())

    finally:
        engine.quit()


if __name__ == "__main__":
    main()
