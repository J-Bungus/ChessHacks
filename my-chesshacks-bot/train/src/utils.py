import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import chess 
import chess.engine
import numpy as np
import random

BOARD_VEC_SIZE = 13  # 6 piece types per color + 1 for empty
NUM_PIECE_TYPES = 6
PIECE_TYPES_DICT = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def encode_board(board: chess.Board):
    """
        Encode board as integer indices (0-12) for nn.Embedding.
        0-5: white pieces, 6-11: black pieces, 12: empty square
        Output shape: (64,)
    """
    enc = torch.zeros(64, dtype=torch.long)
    for square in chess.SQUARES:        
        piece = board.piece_at(square)
        if piece is None:
            try:
                enc[square] = 12  # empty square
            except Exception as e:
                print(f"|{e}|")
        else:
            base_idx = PIECE_TYPES_DICT[piece.piece_type]
            if piece.color == chess.BLACK:
                base_idx += NUM_PIECE_TYPES

                enc[square] = base_idx


    return enc  # (64,)

def encode_extra_state(board: chess.Board):
    """
    Encode extra state as integer indices for embedding (if desired, could use same embedding approach)
    Shape: (5,)
    """
    return torch.tensor([
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
        int(board.turn == chess.WHITE)
    ], dtype=torch.long)  # (5,)


def encode_piece_history(history_boards):
    hist = [encode_board(b) for b in history_boards]
    return hist  # (64, 96)

# ---- Move encoding utilities: 4672 legal UCI moves ----
ALL_SQUARES = [f"{f}{r}" for f in "abcdefgh" for r in "12345678"]
ALL_MOVES = []

for s1 in ALL_SQUARES:
    for s2 in ALL_SQUARES:
        if s1 == s2:
            continue

        ALL_MOVES.append(s1 + s2)  # quiet + capture

        # promotions
        if s1[1] == "7" and s2[1] == "8":
            for promo in ["q", "r", "b", "n"]:
                ALL_MOVES.append(s1 + s2 + promo)
        elif s1[1] == "2" and s2[1] == "1":
            for promo in ["q", "r", "b", "n"]:
                ALL_MOVES.append(s1 + s2 + promo)

MOVE_TO_IDX = {m: i for i, m in enumerate(ALL_MOVES)}
NUM_MOVES = len(ALL_MOVES)
HISTORY_LEN = 1           # current + last 7 positions
NUM_TOKENS  = 64          # 8x8 squares
PIECE_DIM   = 12          # 6 white + 6 black
ANC_DIM     = 9         # ancillary / global info
FEATURE_DIM = HISTORY_LEN * PIECE_DIM + ANC_DIM  # 96 + 16 = 112


def canonicalize_board(board: chess.Board, flip: bool) -> chess.Board:
    """
    If flip=True, mirror the board so that the side to move becomes 'white'
    from the model's point of view.
    Uses python-chess Board.mirror() which swaps colors and flips vertically.
    """
    if not flip:
        return board
    b = board.mirror()
    return b

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]

def piece_to_index(piece: chess.Piece) -> int:
    """
    Map a python-chess Piece to [0..11]:
    0..5: white P N B R Q K
    6..11: black p n b r q k
    """
    if piece is None:
        return -1
    base = PIECE_TYPES.index(piece.piece_type)  # 0..5
    if piece.color == chess.WHITE:
        return base
    else:
        return 6 + base

def encode_piece_history(history_boards, flip: bool) -> np.ndarray:
    """
    history_boards: list[Board] length = HISTORY_LEN, oldest..newest
    flip: whether to canonicalize for side-to-move
    Returns: np.ndarray (64, 96) = (squares, 8*12)
    """
    # Convert all boards to canonical orientation
    canon_boards = [canonicalize_board(b, flip) for b in history_boards]

    # (HISTORY_LEN, 64, 12)
    hist = np.zeros((HISTORY_LEN, NUM_TOKENS, PIECE_DIM), dtype=np.float32)

    for t, b in enumerate(canon_boards):
        # squares 0..63 are a1..h8 in python-chess
        for sq in range(64):
            piece = b.piece_at(sq)
            idx = piece_to_index(piece)
            if idx >= 0:
                hist[t, sq, idx] = 1.0

    # reshape to (64, 8*12)
    hist = hist.transpose(1, 0, 2).reshape(NUM_TOKENS, HISTORY_LEN * PIECE_DIM)
    return hist  # (64, 96)

def encode_piece_history(history_boards, flip: bool) -> np.ndarray:
    """
    history_boards: list[Board] length = HISTORY_LEN, oldest..newest
    flip: whether to canonicalize for side-to-move
    Returns: np.ndarray (64, 96) = (squares, 8*12)
    """
    # Convert all boards to canonical orientation
    canon_boards = [canonicalize_board(b, flip) for b in history_boards]

    # (HISTORY_LEN, 64, 12)
    hist = np.zeros((HISTORY_LEN, NUM_TOKENS, PIECE_DIM), dtype=np.float32)

    for t, b in enumerate(canon_boards):
        # squares 0..63 are a1..h8 in python-chess
        for sq in range(64):
            piece = b.piece_at(sq)
            idx = piece_to_index(piece)
            if idx >= 0:
                hist[t, sq, idx] = 1.0

    # reshape to (64, 8*12)
    hist = hist.transpose(1, 0, 2).reshape(NUM_TOKENS, HISTORY_LEN * PIECE_DIM)
    return hist  # (64, 96)

def encode_ancillary_features(current_board: chess.Board,
                              flip: bool,
                              rep_flags: np.ndarray) -> np.ndarray:
    """
    current_board: Board at time t (original orientation)
    flip: whether side-to-move is black in original board
    rep_flags: np.ndarray (HISTORY_LEN,) of 0/1 repetition flags
               aligned oldest..newest for the history window

    Returns: (16,) float32
    """
    # Canonicalize current board to side-to-move POV
    b = canonicalize_board(current_board, flip)

    # 4 castling rights (now from canonical POV)
    castle_feats = np.array([
        float(b.has_kingside_castling_rights(chess.WHITE)),
        float(b.has_queenside_castling_rights(chess.WHITE)),
        float(b.has_kingside_castling_rights(chess.BLACK)),
        float(b.has_queenside_castling_rights(chess.BLACK)),
    ], dtype=np.float32)

    # En passant
    ep_exists = 0.0
    ep_file_norm = 0.0
    if b.ep_square is not None:
        ep_exists = 1.0
        file_idx = chess.square_file(b.ep_square)  # 0..7
        ep_file_norm = file_idx / 7.0

    ep_feats = np.array([ep_exists, ep_file_norm], dtype=np.float32)

    # Rule-50
    rule50 = b.halfmove_clock / 100.0
    rule50_feats = np.array([rule50], dtype=np.float32)

    # Repetition flags (truncate/pad to HISTORY_LEN)
    rep_flags = np.asarray(rep_flags, dtype=np.float32)
    if rep_flags.shape[0] != HISTORY_LEN:
        # safety, though we'll pass correct size
        rep_pad = np.zeros(HISTORY_LEN, dtype=np.float32)
        rep_pad[-rep_flags.shape[0]:] = rep_flags
        rep_flags = rep_pad

    # Dummy bias bit
    bias = np.array([1.0], dtype=np.float32)

    anc = np.concatenate([castle_feats, ep_feats, rule50_feats, rep_flags, bias])
    assert anc.shape[0] == ANC_DIM
    return anc  # (16,)

def encode_history_to_tokens(history_boards, rep_flags) -> torch.Tensor:
    """
    history_boards: list[Board] length = HISTORY_LEN, oldest..newest
    rep_flags: list/array length = HISTORY_LEN, 0/1 repetition flags
               for those same boards.
    Returns: torch.Tensor (64, 112)
    """
    assert len(history_boards) == HISTORY_LEN
    assert len(rep_flags) == HISTORY_LEN

    current_board = history_boards[-1]
    flip = (current_board.turn == chess.BLACK)

    piece_hist = encode_piece_history(history_boards, flip)   # (64, 96)
    anc = encode_ancillary_features(current_board, flip, np.array(rep_flags))  # (16,)

    # broadcast anc to all 64 squares
    anc_broadcast = np.repeat(anc[None, :], NUM_TOKENS, axis=0)  # (64, 16)

    feats = np.concatenate([piece_hist, anc_broadcast], axis=1)  # (64, 112)
    return torch.tensor(feats, dtype=torch.float32)

def play_self_play_game(
    engine: chess.engine.SimpleEngine,
    max_moves: int = 200,
    nodes: int = 300,
    multipv: int = 8,
    epsilon: float = 0.10,   # Dirichlet noise weight
    alpha: float = 0.3,      # Dirichlet concentration
):
    """
    Self-play with exploration.

    Returns:
      positions: list[Board] states BEFORE each move
      moves:     list[chess.Move]
      evals_cp:  list[float] evals in centipawns from side-to-move POV
    """
    board = chess.Board()
    positions = []
    moves = []
    evals_cp = []

    while not board.is_game_over() and len(moves) < max_moves:
        positions.append(board.copy(stack=False))

        # Ask engine for several candidate moves
        info = engine.analyse(
            board,
            chess.engine.Limit(nodes=nodes),
            multipv=multipv,
        )

        if multipv == 1:
            info = [info]

        candidate_moves = []
        candidate_scores = []

        for line in info:
            if "pv" not in line or not line["pv"]:
                continue
            mv = line["pv"][0]
            score_white = line["score"].white().score(mate_score=100000)
            if score_white is None:
                continue

            # convert to side-to-move eval in cp
            stm_eval_cp = score_white if board.turn == chess.WHITE else -score_white

            candidate_moves.append(mv)
            candidate_scores.append(stm_eval_cp)

        # Fallback if engine didn't give multipv lines for some reason
        if not candidate_moves:
            legal = list(board.legal_moves)
            if not legal:
                break
            mv = random.choice(legal)
            moves.append(mv)
            evals_cp.append(0.0)
            board.push(mv)
            continue

        # Save eval for *current* position (using the best candidate as reference)
        # Here we use the highest-scoring candidate as the eval cp
        best_idx = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
        best_eval_cp = candidate_scores[best_idx]
        evals_cp.append(best_eval_cp)

        # Turn scores into a probability distribution over candidate moves
        scores_tensor = torch.tensor(candidate_scores, dtype=torch.float32)

        # Temperature can be tuned; 400.0 ≈ 1 pawn
        logits = scores_tensor / 400.0
        probs = F.softmax(logits, dim=0)

        # Add Dirichlet noise for exploration
        noise = torch.distributions.Dirichlet(alpha * torch.ones_like(probs)).sample()
        probs = (1.0 - epsilon) * probs + epsilon * noise

        # Sample a move index from this distribution
        idx = torch.multinomial(probs, 1).item()
        mv = candidate_moves[idx]

        moves.append(mv)
        board.push(mv)

    return positions, moves, evals_cp

def eval_cp_to_target(stm_eval_cp: float) -> float:
    # cp ~400 ≈ 1 pawn advantage → ~0.76 after tanh(1)
    return float(np.tanh(stm_eval_cp / 400.0))

def compute_repetition_flags(positions):
    """
    positions: list[Board] BEFORE each move, in order.
    Returns:
      rep_flags_all: list[np.ndarray] of shape (HISTORY_LEN,)
                     for each index i in positions.
    """
    fen_counts = {}
    fens = []
    for b in positions:
        f = b.board_fen()  # ignore clocks for repetition purposes
        fens.append(f)
        fen_counts[f] = fen_counts.get(f, 0) + 1

    rep_flags_all = []

    for i in range(len(positions)):
        # history indices for this window: oldest..current
        start = max(0, i - (HISTORY_LEN - 1))
        hist_idx = list(range(start, i + 1))

        # pad with "non-repetition" at front if needed
        flags = []
        pad_len = HISTORY_LEN - len(hist_idx)
        flags.extend([0] * pad_len)

        for j in hist_idx:
            f = fens[j]
            flags.append(1 if fen_counts[f] >= 2 else 0)

        rep_flags_all.append(np.array(flags, dtype=np.int64))

    return rep_flags_all

def get_history_stack(positions, idx):
    """
    positions: list[Board]
    idx: index of current position
    Returns: list[Board] length HISTORY_LEN, oldest..newest
    """
    start = max(0, idx - (HISTORY_LEN - 1))
    hist = positions[start:idx+1]

    if len(hist) < HISTORY_LEN:
        pad_count = HISTORY_LEN - len(hist)
        empty = chess.Board()
        empty.clear()
        hist = [empty] * pad_count + hist

    return hist
