import numpy as np
import chess

PIECE_VAL = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
}

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5}
EXTENDED_CENTER = {
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
}


def basic_feature_names() -> list[str]:
    return [
        "is_capture",
        "gives_check",
        "is_promotion",
        "piece_type",
        "to_square",
        "material_balance",
        "mobility",
        "side_to_move",
    ]


def candidate_feature_names() -> list[str]:
    return [
        "is_capture",
        "gives_check",
        "is_promotion",
        "is_castling",
        "is_en_passant",
        "piece_type",
        "captured_piece_type",
        "from_file",
        "from_rank",
        "to_file",
        "to_rank",
        "move_file_delta",
        "move_rank_delta",
        "to_center_4",
        "to_center_16",
        "material_balance",
        "mobility_before",
        "engine_score_cp",
        "score_gap_from_best_cp",
        "engine_rank",
        "engine_rank_norm",
        "candidate_count",
        "side_to_move",
    ]


def material_balance(board: chess.Board) -> int:
    """White material - Black material (simple piece values)."""
    s = 0
    for p, v in PIECE_VAL.items():
        s += v * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK)))
    return s


def featurize_move(board: chess.Board, move: chess.Move) -> np.ndarray:
    """
    Legacy 8-dim feature vector kept for compatibility with already-trained models.
    """
    is_cap = int(board.is_capture(move))
    gives_check = int(board.gives_check(move))
    is_prom = int(move.promotion is not None)

    piece = board.piece_at(move.from_square)
    piece_type = piece.piece_type if piece else 0

    to_sq = move.to_square
    mat = material_balance(board)
    mob = board.legal_moves.count()
    stm = int(board.turn)  # 1 if white to move, 0 if black to move

    return np.array([is_cap, gives_check, is_prom, piece_type, to_sq, mat, mob, stm], dtype=np.float32)


def featurize_candidate(
    board: chess.Board,
    move: chess.Move,
    engine_score_cp: float,
    best_score_cp: float,
    rank: int,
    candidate_count: int,
) -> np.ndarray:
    """
    Richer feature vector for top-k move-choice training/inference.
    Includes move geometry and engine metadata (score/rank/gap).
    """
    is_cap = int(board.is_capture(move))
    gives_check = int(board.gives_check(move))
    is_prom = int(move.promotion is not None)
    is_castling = int(board.is_castling(move))
    is_ep = int(board.is_en_passant(move))

    piece = board.piece_at(move.from_square)
    piece_type = piece.piece_type if piece else 0

    cap_piece = board.piece_at(move.to_square)
    if cap_piece is not None:
        captured_piece_type = cap_piece.piece_type
    elif is_ep:
        captured_piece_type = chess.PAWN
    else:
        captured_piece_type = 0

    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    move_file_delta = abs(to_file - from_file)
    move_rank_delta = abs(to_rank - from_rank)

    to_center_4 = int(move.to_square in CENTER_SQUARES)
    to_center_16 = int(move.to_square in EXTENDED_CENTER)

    mat = material_balance(board)
    mob = board.legal_moves.count()
    stm = int(board.turn)

    gap_from_best = float(best_score_cp - engine_score_cp)
    rank_norm = float(rank / max(candidate_count - 1, 1))

    return np.array(
        [
            is_cap,
            gives_check,
            is_prom,
            is_castling,
            is_ep,
            piece_type,
            captured_piece_type,
            from_file,
            from_rank,
            to_file,
            to_rank,
            move_file_delta,
            move_rank_delta,
            to_center_4,
            to_center_16,
            mat,
            mob,
            float(np.clip(engine_score_cp, -2000, 2000)),
            float(np.clip(gap_from_best, 0, 2000)),
            float(rank),
            rank_norm,
            float(candidate_count),
            stm,
        ],
        dtype=np.float32,
    )


def model_feature_dim(model) -> int | None:
    """Best-effort feature dimension detection for backward compatibility."""
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    if hasattr(model, "coef_"):
        return int(model.coef_.shape[1])
    return None


def featurize_for_model(
    model,
    board: chess.Board,
    move: chess.Move,
    engine_score_cp: float,
    best_score_cp: float,
    rank: int,
    candidate_count: int,
) -> np.ndarray:
    """
    Produce features matching the expected model input size.
    Uses the richer candidate featurizer for new models and legacy 8-dim features for old ones.
    """
    dim = model_feature_dim(model)
    rich = featurize_candidate(board, move, engine_score_cp, best_score_cp, rank, candidate_count)
    if dim is None or dim == rich.shape[0]:
        return rich

    basic = featurize_move(board, move)
    if dim == basic.shape[0]:
        return basic

    raise ValueError(
        f"Model expects {dim} features, but available featurizers provide "
        f"{basic.shape[0]} or {rich.shape[0]}."
    )


def position_key(board: chess.Board) -> str:
    """Stable position key without move clocks, for opening-book lookup."""
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    turn = "w" if board.turn == chess.WHITE else "b"
    return f"{board.board_fen()} {turn} {board.castling_xfen()} {ep}"


def pick_book_move(
    bundle: dict,
    board: chess.Board,
    cand: list[tuple[chess.Move, float]],
    max_loss_cp: float = 80.0,
):
    """
    Return a book move if the position exists in bundle['opening_book'] and
    the move is present in candidate list within max_loss_cp from engine best.
    """
    if not isinstance(bundle, dict):
        return None, None
    book = bundle.get("opening_book")
    if not isinstance(book, dict):
        return None, None

    entry = book.get(position_key(board))
    if not entry:
        return None, None

    cand_map = {mv.uci(): (mv, sc) for mv, sc in cand}
    best_score = cand[0][1]

    for m in entry.get("moves", []):
        uci = m.get("uci")
        if uci not in cand_map:
            continue
        mv, sc = cand_map[uci]
        gap = float(best_score - sc)
        if gap <= max_loss_cp:
            details = {
                "uci": uci,
                "freq": float(m.get("freq", 0.0)),
                "count": int(m.get("count", 0)),
                "gap_cp": gap,
                "total": int(entry.get("total", 0)),
            }
            return mv, details

    return None, None


def pick_memory_move(bundle: dict, board: chess.Board):
    """
    Return the highest-frequency memorized move for this exact position.
    This is pure imitation: no engine filter, only legality check.
    """
    if not isinstance(bundle, dict):
        return None, None
    mem = bundle.get("position_memory")
    if not isinstance(mem, dict):
        return None, None

    entry = mem.get(position_key(board))
    if not entry:
        return None, None

    for m in entry.get("moves", []):
        uci = m.get("uci")
        if not isinstance(uci, str):
            continue
        try:
            mv = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if mv in board.legal_moves:
            details = {
                "uci": uci,
                "count": int(m.get("count", 0)),
                "freq": float(m.get("freq", 0.0)),
                "total": int(entry.get("total", 0)),
            }
            return mv, details

    return None, None
