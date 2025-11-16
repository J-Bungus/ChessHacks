import torch
from torch.utils.data import Dataset
import chess
import chess.engine
import numpy as np
import random

from tqdm import tqdm
from utils import encode_board, encode_extra_state, MOVE_TO_IDX



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

def get_history_stack(history_len, positions, idx):
    start = max(0, idx - (history_len - 1))
    hist = positions[start:idx+1]

    if len(hist) < history_len:
        pad_count = history_len - len(hist)
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
                 movetime=0.01,
                 history_length=1):

        self.samples = []

        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        for _ in tqdm(range(num_games)):
            positions, moves, evals_cp = play_self_play_game(
                engine,
                max_moves=max_moves,
                movetime=movetime
            )

            if len(moves) == 0:
                continue

            # rep_flags_all = compute_repetition_flags(positions)

            for i, (board, move, eval_cp) in enumerate(zip(positions, moves, evals_cp)):
                # history_boards = get_history_stack(history_length, positions, i)
                # rep_flags = rep_flags_all[i]

                board_tokens = encode_board(board)
                state = encode_extra_state(board)

                u = move.uci()
                if u not in MOVE_TO_IDX:
                    continue
                policy_idx = MOVE_TO_IDX[u]
                

                # value: static eval from side-to-move POV, in [-1,1]
                v = eval_cp_to_target(eval_cp)
                v = torch.tensor(v, dtype=torch.float32)

                self.samples.append((board_tokens, state, policy_idx, v))

        engine.quit()
        print(f"Generated {len(self.samples)} samples from {num_games} games.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
