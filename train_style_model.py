#!/usr/bin/env python3
# ^ Tells your OS: “run this file using python3” (mainly useful on Linux/macOS)

import argparse
# ^ Lets you read command-line arguments like: --target_pgn magnus.pgn

import numpy as np
# ^ Numerical arrays + random sampling (balancing dataset)

import chess
# ^ Core chess objects (board, colors, pieces)

import chess.pgn
# ^ Reads chess games from PGN files

import joblib
# ^ Saves/loads Python objects (your trained model bundle)

from sklearn.linear_model import LogisticRegression
# ^ The ML model (a linear classifier that outputs probabilities)

from sklearn.model_selection import train_test_split
# ^ Splits dataset into train/test

from sklearn.metrics import classification_report, roc_auc_score
# ^ Prints accuracy/precision/recall/f1; computes AUC

from style_ranker import featurize_move
# ^ Your feature extractor: converts (board, move) -> numeric feature vector


def load_target_moves(pgn_path: str, target_substr: str, max_games: int, max_plies: int):
    """
    Collect ONLY the moves played by the target player (Magnus),
    based on whether target_substr appears in PGN headers.
    Label = 1 for these moves.
    """
    X, y = [], []  # X = feature vectors, y = labels (1 = target)
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        # ^ Open PGN file; ignore broken characters

        for _ in range(max_games):
            # ^ Read up to max_games games
            game = chess.pgn.read_game(f)
            # ^ Parse one game from the file
            if game is None:
                # ^ No more games in file
                break

            white = game.headers.get("White", "")
            # ^ Player name string for White from PGN headers
            black = game.headers.get("Black", "")
            # ^ Player name string for Black from PGN headers

            board = game.board()
            # ^ Start from initial chess position
            ply = 0
            # ^ Move counter inside this game (half-moves)

            for mv in game.mainline_moves():
                # ^ Iterate through moves in the main line of the PGN
                if ply >= max_plies:
                    # ^ Stop early so one game doesn’t dominate
                    break
                if mv not in board.legal_moves:
                    # ^ Safety: PGN may contain illegal move for the board state
                    break

                is_target_move = (
                    (board.turn == chess.WHITE and target_substr in white) or
                    (board.turn == chess.BLACK and target_substr in black)
                )
                # ^ True if it is the target player’s turn right now

                if is_target_move:
                    # ^ Only collect moves made by the target player
                    X.append(featurize_move(board, mv))
                    # ^ Convert this candidate move into a numeric vector
                    y.append(1)
                    # ^ Label “1 = target style”

                board.push(mv)
                # ^ Apply the move to advance the game
                ply += 1
                # ^ Increment half-move counter

    if not X:
        # ^ If we never collected anything, training would be impossible
        raise RuntimeError("No target moves loaded. Check PGN and target name substring.")
    return np.stack(X), np.array(y)
    # ^ Return X as matrix (N x D), y as vector (N,)


def load_all_moves_as_negative(pgn_path: str, max_games: int, max_plies: int):
    """
    Collect moves from other players.
    Label = 0 for these moves.
    """
    X, y = [], []  # X = features, y = labels (0 = non-target)
    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        # ^ Open the “others” PGN

        for _ in range(max_games):
            # ^ Read up to max_games
            game = chess.pgn.read_game(f)
            # ^ Parse a game
            if game is None:
                # ^ End of file
                break

            board = game.board()
            # ^ Reset board for each game
            ply = 0
            # ^ Half-move counter

            for mv in game.mainline_moves():
                # ^ Loop moves
                if ply >= max_plies:
                    break
                if mv not in board.legal_moves:
                    break

                X.append(featurize_move(board, mv))
                # ^ Extract features for *other* players’ moves
                y.append(0)
                # ^ Label “0 = non-target”

                board.push(mv)
                # ^ Advance board
                ply += 1

    if not X:
        raise RuntimeError("No negative moves loaded. Check others.pgn.")
    return np.stack(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    # ^ Creates a CLI argument parser

    ap.add_argument("--target_pgn", required=True)
    # ^ Path to target PGN (Magnus games)

    ap.add_argument("--others_pgn", required=True)
    # ^ Path to “others” PGN (everyone else)

    ap.add_argument("--target_name", default="Carlsen",
                    help='substring to identify target in headers, e.g. "Carlsen"')
    # ^ Substring match in PGN headers; default is Carlsen

    ap.add_argument("--out", default="style_model.pkl")
    # ^ Where to save the trained model

    ap.add_argument("--max_games", type=int, default=400)
    # ^ Limit to avoid huge runtime

    ap.add_argument("--max_plies", type=int, default=120)
    # ^ Limit per game to avoid one long game dominating

    args = ap.parse_args()
    # ^ Actually parse the CLI args

    X_pos, y_pos = load_target_moves(args.target_pgn, args.target_name,
                                     args.max_games, args.max_plies)
    # ^ Positive class: only target’s moves

    X_neg, y_neg = load_all_moves_as_negative(args.others_pgn,
                                              args.max_games, args.max_plies)
    # ^ Negative class: everyone else’s moves

    # Balance dataset by downsampling larger class to smaller class size
    n = min(len(y_pos), len(y_neg))
    # ^ Make both classes equal size so the classifier doesn’t “cheat” by always guessing the big class

    rng = np.random.default_rng(42)
    # ^ Reproducible randomness

    pos_idx = rng.choice(len(y_pos), size=n, replace=False)
    # ^ Randomly pick n positives
    neg_idx = rng.choice(len(y_neg), size=n, replace=False)
    # ^ Randomly pick n negatives

    X = np.vstack([X_pos[pos_idx], X_neg[neg_idx]])
    # ^ Combine into one big feature matrix
    y = np.concatenate([y_pos[pos_idx], y_neg[neg_idx]])
    # ^ Combine labels

    # Shuffle
    perm = rng.permutation(len(y))
    # ^ Random permutation of indices
    X, y = X[perm], y[perm]
    # ^ Shuffle X and y in the same way

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # ^ Split 80/20 with same class ratio in both sets

    clf = LogisticRegression(max_iter=5000)
    # ^ Create model; max_iter higher to ensure convergence
    clf.fit(X_train, y_train)
    # ^ Train logistic regression weights

    probs = clf.predict_proba(X_test)[:, 1]
    # ^ Probability “this move looks like the target”
    auc = roc_auc_score(y_test, probs)
    # ^ AUC: 0.5 random, 1.0 perfect

    print("\n=== Test report ===")
    print(classification_report(y_test, clf.predict(X_test), digits=3))
    print(f"AUC: {auc:.3f}")

    bundle = {
        "model": clf,
        "target_name": args.target_name,
        "features": [
            "is_capture", "gives_check", "is_promotion", "piece_type",
            "to_square", "material_balance", "mobility", "side_to_move"
        ]
    }
    # ^ “Bundle” = model + metadata you want to remember

    joblib.dump(bundle, args.out)
    # ^ Save bundle to disk
    print(f"\nSaved model to: {args.out}")


if __name__ == "__main__":
    # ^ Only run main() if you executed this file directly
    main()
