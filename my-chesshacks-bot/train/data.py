import torch
from torch.utils.data import Dataset
import chess
import chess.engine
import numpy as np
import random


# ---- Move encoding utilities: 4672 legal UCI moves ----
ALL_SQUARES = [f"{f}{r}" for f in "abcdefgh" for r in "12345678"]
ALL_MOVES = []

# quiet moves + captures + promotions
for s1 in ALL_SQUARES:
    for s2 in ALL_SQUARES:
        ALL_MOVES.append(s1 + s2)
        for promo in ["q", "r", "b", "n"]:
            ALL_MOVES.append(s1 + s2 + promo)

MOVE_TO_IDX = {m: i for i, m in enumerate(ALL_MOVES)}
NUM_MOVES = len(ALL_MOVES)   # 4672


class ChessPolicyDataset(Dataset):
    """
    Generates (18×8×8 board planes, best_move_index)
    compatible with ChessPolicyModel.
    """

    def __init__(self,
                 n_positions=5000,
                 max_random_moves=20,
                 stockfish_path="/usr/bin/stockfish",
                 stockfish_depth=12):

        self.n_positions = n_positions
        self.max_random_moves = max_random_moves
        self.depth = stockfish_depth

        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.samples = []

        for _ in range(n_positions):
            board = self.random_position(max_random_moves)

            best_move = self.get_best_move(board)
            if best_move is None:
                continue

            x = self.encode_18_planes(board)
            y = MOVE_TO_IDX[best_move.uci()]

            self.samples.append((x, y))

        self.engine.quit()

    # ----------------------------------------------------------
    # Generate random position by randomly advancing the game
    # ----------------------------------------------------------
    def random_position(self, max_moves):
        b = chess.Board()
        for _ in range(random.randint(0, max_moves)):
            if b.is_game_over():
                break
            move = random.choice(list(b.legal_moves))
            b.push(move)
        return b

    # ----------------------------------------------------------
    # Stockfish best move
    # ----------------------------------------------------------
    def get_best_move(self, board):
        info = self.engine.analyse(
            board,
            limit=chess.engine.Limit(depth=self.depth)
        )
        return info["pv"][0] if "pv" in info else None

    # ----------------------------------------------------------
    # 18-plane encoding to match your model
    # ----------------------------------------------------------
    def encode_18_planes(self, board):
        planes = []

        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                    chess.ROOK, chess.QUEEN, chess.KING]

        # 12 planes
        for piece_type in piece_types:
            p_white = np.zeros((8, 8), np.float32)
            for sq in board.pieces(piece_type, chess.WHITE):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                p_white[r, c] = 1.0
            planes.append(p_white)

            p_black = np.zeros((8, 8), np.float32)
            for sq in board.pieces(piece_type, chess.BLACK):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                p_black[r, c] = 1.0
            planes.append(p_black)

        # 1 plane: side to move
        stm = np.ones((8, 8), np.float32) if board.turn == chess.WHITE else np.zeros((8, 8), np.float32)
        planes.append(stm)

        # 4 planes: castling rights
        castling_features = [
            board.has_kingside_castling_rights(chess.WHITE),
            board.has_queenside_castling_rights(chess.WHITE),
            board.has_kingside_castling_rights(chess.BLACK),
            board.has_queenside_castling_rights(chess.BLACK),
        ]

        for flag in castling_features:
            planes.append(np.ones((8,8), np.float32) if flag else np.zeros((8,8), np.float32))

        # 1 extra plane: zeros to reach 18 planes
        planes.append(np.zeros((8,8), np.float32))

        assert len(planes) == 18, f"Expected 18 planes, got {len(planes)}"

        arr = np.stack(planes)
        return torch.tensor(arr, dtype=torch.float32)


    # ----------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
