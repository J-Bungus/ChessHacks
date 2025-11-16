import torch
from torch.utils.data import Dataset
import chess
import chess.engine
import numpy as np
import random

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

def play_self_play_game(engine, max_moves=200, movetime=0.01):
    """
    Returns:
      positions: list[Board] states BEFORE each move
      moves:     list[chess.Move]
      evals:     list[float] cp evals from side-to-move's POV
    """
    board = chess.Board()
    positions = []
    moves = []
    evals = []

    while not board.is_game_over() and len(moves) < max_moves:
        positions.append(board.copy(stack=False))

        info = engine.analyse(board, chess.engine.Limit(time=movetime))

        score_white = info["score"].white().score(mate_score=100000)
        if score_white is None:
            break  # avoid weird mate/NaN

        # we want eval from *side-to-move* POV
        # engine gives score from white's POV
        if board.turn == chess.WHITE:
            stm_eval_cp = score_white
        else:
            stm_eval_cp = -score_white

        evals.append(stm_eval_cp)

        pv = info.get("pv")
        if not pv:
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = random.choice(legal_moves)
        else:
            move = pv[0]

        moves.append(move)
        board.push(move)

    return positions, moves, evals

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

class SelfPlayDataset(Dataset):
    """
    Outputs:
      x: (64, 112) tokens for transformer input
      policy_target: integer move index in [0, NUM_MOVES)
      value_target: scalar in [-1, 1], Stockfish eval from side-to-move POV
    """

    def __init__(self,
                 num_games=50,
                 max_moves=200,
                 stockfish_path="/usr/bin/stockfish",
                 movetime=0.01):

        self.samples = []

        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        for _ in range(num_games):
            positions, moves, evals_cp = play_self_play_game(
                engine,
                max_moves=max_moves,
                movetime=movetime
            )

            if len(moves) == 0:
                continue

            rep_flags_all = compute_repetition_flags(positions)

            for i, (board, move, eval_cp) in enumerate(zip(positions, moves, evals_cp)):
                history_boards = get_history_stack(positions, i)
                rep_flags = rep_flags_all[i]

                x = encode_history_to_tokens(history_boards, rep_flags)  # (64, 112)

                u = move.uci()
                if u not in MOVE_TO_IDX:
                    continue
                policy_idx = MOVE_TO_IDX[u]

                # value: static eval from side-to-move POV, in [-1,1]
                v = eval_cp_to_target(eval_cp)
                v = torch.tensor(v, dtype=torch.float32)

                self.samples.append((x, policy_idx, v))

        engine.quit()
        print(f"Generated {len(self.samples)} samples from {num_games} games.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, p, v = self.samples[idx]
        return x, p, v