#!/usr/bin/env python3
import argparse
from collections import Counter

import numpy as np

import chess
import chess.pgn
import chess.engine

import joblib

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from style_ranker import candidate_feature_names, featurize_candidate, position_key


def parse_lam_grid(raw: str) -> list[float]:
    vals = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("lam grid is empty")
    return sorted(set(vals))


def get_topk(engine, board, k, depth):
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=k)
    cand = []
    for item in info:
        mv = item["pv"][0]
        score = item["score"].pov(board.turn).score(mate_score=100000)
        cand.append((mv, score))
    cand.sort(key=lambda t: t[1], reverse=True)
    return cand


def evaluate_by_position(y, probs, engine_scores, groups, lam_grid):
    group_to_rows = {}
    for i, g in enumerate(groups):
        group_to_rows.setdefault(int(g), []).append(i)

    total = 0
    engine_match = 0
    model_match = 0
    lam_match = {lam: 0 for lam in lam_grid}
    lam_cp_loss = {lam: [] for lam in lam_grid}

    for row_ids in group_to_rows.values():
        row_ids = np.array(row_ids, dtype=np.int64)
        y_g = y[row_ids]
        if y_g.sum() != 1:
            continue

        p_g = probs[row_ids]
        s_g = engine_scores[row_ids]
        true_idx = int(np.argmax(y_g))
        best_engine_idx = int(np.argmax(s_g))

        total += 1
        if best_engine_idx == true_idx:
            engine_match += 1
        if int(np.argmax(p_g)) == true_idx:
            model_match += 1

        best_engine_score = float(s_g[best_engine_idx])
        for lam in lam_grid:
            vals = s_g + lam * p_g
            pick_idx = int(np.argmax(vals))
            if pick_idx == true_idx:
                lam_match[lam] += 1
            lam_cp_loss[lam].append(best_engine_score - float(s_g[pick_idx]))

    metrics = {
        "positions": total,
        "engine_top1_match": (engine_match / total) if total else 0.0,
        "model_top1_match": (model_match / total) if total else 0.0,
        "lam": {},
    }
    for lam in lam_grid:
        if lam_cp_loss[lam]:
            losses = np.array(lam_cp_loss[lam], dtype=np.float32)
            avg_loss = float(np.mean(losses))
            med_loss = float(np.median(losses))
        else:
            avg_loss = 0.0
            med_loss = 0.0

        metrics["lam"][lam] = {
            "match": (lam_match[lam] / total) if total else 0.0,
            "avg_cp_loss": avg_loss,
            "median_cp_loss": med_loss,
        }
    return metrics


def choose_best_lam(lam_metrics: dict[float, dict[str, float]]) -> float:
    # Prefer highest match; on ties prefer lower cp loss, then smaller lambda.
    best = None
    for lam, m in lam_metrics.items():
        key = (m["match"], -m["avg_cp_loss"], -lam)
        if best is None or key > best[0]:
            best = (key, lam)
    return float(best[1])


def build_opening_book(raw_counts, min_count: int, top_n: int):
    book = {}
    for pos_key, counter in raw_counts.items():
        total = int(sum(counter.values()))
        if total < min_count:
            continue

        moves = []
        for uci, count in counter.most_common(top_n):
            if count < min_count:
                continue
            moves.append(
                {
                    "uci": uci,
                    "count": int(count),
                    "freq": float(count / total),
                }
            )

        if moves:
            book[pos_key] = {
                "total": total,
                "moves": moves,
            }

    return book


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_pgn", required=True)
    ap.add_argument("--target_name", default="Carlsen")
    ap.add_argument("--stockfish", required=True)
    ap.add_argument("--out", default="choice_model.pkl")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--max_games", type=int, default=300)
    ap.add_argument("--max_plies", type=int, default=120)
    ap.add_argument("--model", choices=["hgb", "logreg"], default="hgb")
    ap.add_argument("--lam_grid", default="0,25,50,75,100,150,200,250,400,600,900")
    ap.add_argument("--book_max_plies", type=int, default=24)
    ap.add_argument("--book_min_count", type=int, default=2)
    ap.add_argument("--book_top_n", type=int, default=3)
    ap.add_argument("--memory_min_count", type=int, default=1)
    ap.add_argument("--memory_top_n", type=int, default=1)
    args = ap.parse_args()

    lam_grid = parse_lam_grid(args.lam_grid)

    X, y = [], []
    groups = []
    engine_scores = []
    raw_book_counts = {}
    raw_memory_counts = {}
    position_id = 0
    total_target_positions = 0
    topk_covered_positions = 0

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
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

                    if is_target_turn:
                        total_target_positions += 1

                        mem_key = position_key(board)
                        mem_counter = raw_memory_counts.setdefault(mem_key, Counter())
                        mem_counter[true_mv.uci()] += 1

                        if ply < args.book_max_plies:
                            book_counter = raw_book_counts.setdefault(mem_key, Counter())
                            book_counter[true_mv.uci()] += 1

                        cand = get_topk(engine, board, args.k, args.depth)
                        cand_moves = [mv for mv, _ in cand]

                        if true_mv in cand_moves:
                            topk_covered_positions += 1
                            best_sc = cand[0][1]
                            n_cand = len(cand)
                            for rank, (mv, sc) in enumerate(cand):
                                X.append(featurize_candidate(board, mv, sc, best_sc, rank, n_cand))
                                y.append(1 if mv == true_mv else 0)
                                groups.append(position_id)
                                engine_scores.append(float(sc))
                            position_id += 1

                    board.push(true_mv)
                    ply += 1
    finally:
        engine.quit()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups, dtype=np.int64)
    engine_scores = np.array(engine_scores, dtype=np.float32)

    if len(np.unique(y)) < 2:
        raise RuntimeError("Not enough positive examples. Increase --k or lower --depth.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_test = groups[test_idx]
    scores_test = engine_scores[test_idx]

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = (neg / max(pos, 1))
    sample_weight = np.where(y_train == 1, pos_weight, 1.0).astype(np.float32)

    if args.model == "hgb":
        clf = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=0.2,
            random_state=42,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
        clf.fit(X_train, y_train)

    probs_test = clf.predict_proba(X_test)[:, 1]
    row_auc = roc_auc_score(y_test, probs_test)
    row_ap = average_precision_score(y_test, probs_test)
    pos_metrics = evaluate_by_position(y_test, probs_test, scores_test, groups_test, lam_grid)
    best_lam = choose_best_lam(pos_metrics["lam"])
    opening_book = build_opening_book(
        raw_book_counts,
        min_count=args.book_min_count,
        top_n=args.book_top_n,
    )
    position_memory = build_opening_book(
        raw_memory_counts,
        min_count=args.memory_min_count,
        top_n=args.memory_top_n,
    )

    print("\n=== Choice Model v2 ===")
    print(f"Target: {args.target_name}")
    print(f"Rows: {len(y)}")
    print(f"Target positions seen: {total_target_positions}")
    print(
        "Top-k covered positions: "
        f"{topk_covered_positions} "
        f"({topk_covered_positions/max(total_target_positions,1):.3f})"
    )
    print(f"Train rows: {len(y_train)}  Test rows: {len(y_test)}")
    print(f"Row AUC: {row_auc:.3f}")
    print(f"Row AP:  {row_ap:.3f}")
    print(f"Validation positions: {pos_metrics['positions']}")
    print(f"Engine top-1 match: {pos_metrics['engine_top1_match']:.3f}")
    print(f"Model top-1 match:  {pos_metrics['model_top1_match']:.3f}")
    print(f"Opening-book positions: {len(opening_book)}")
    print(f"Position-memory positions: {len(position_memory)}")
    print("\nLambda sweep (validation):")
    for lam in lam_grid:
        lm = pos_metrics["lam"][lam]
        print(f"  lam={lam:>6g}  match={lm['match']:.3f}  avg_cp_loss={lm['avg_cp_loss']:.2f}")
    print(f"\nRecommended lam: {best_lam:g}")

    bundle = {
        "model": clf,
        "type": "choice_v2",
        "target_name": args.target_name,
        "k": args.k,
        "depth": args.depth,
        "features": candidate_feature_names(),
        "feature_version": "candidate_v2",
        "model_kind": args.model,
        "recommended_lam": best_lam,
        "lam_grid": lam_grid,
        "validation": pos_metrics,
        "opening_book": opening_book,
        "opening_book_max_plies": int(args.book_max_plies),
        "opening_book_min_count": int(args.book_min_count),
        "position_memory": position_memory,
        "memory_min_count": int(args.memory_min_count),
        "memory_top_n": int(args.memory_top_n),
        "row_auc": float(row_auc),
        "row_ap": float(row_ap),
    }

    joblib.dump(bundle, args.out)
    print(f"Saved model to: {args.out}")


if __name__ == "__main__":
    main()
