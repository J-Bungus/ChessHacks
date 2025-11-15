import torch
from torch.utils.data import Dataset
import chess
import chess.engine
import random
import numpy as np

class AlphaZeroChessDataset(Dataset):
    """
    Self-contained dataset that:
      - Generates random positions by playing random legal moves.
      - Evaluates each position using Stockfish.
      - Encodes positions using a simplified AlphaZero-style 119-plane representation.
    """

    def __init__(self,
                 n_positions=5000,
                 max_random_moves=20,
                 stockfish_path="/usr/bin/stockfish",
                 stockfish_depth=12):

        self.n_positions = n_positions
        self.max_random_moves = max_random_moves
        self.stockfish_depth = stockfish_depth

        # Stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

        # Pre-generate dataset
        self.samples = []
        for _ in range(n_positions):
            board = self.random_position(max_random_moves)
            value = self.evaluate(board)
            features = self.encode_board(board)
            self.samples.append((features, value))

        # Close engine
        self.engine.quit()

    # ----------------------------------------------------------
    #  Generate random positions
    # ----------------------------------------------------------
    def random_position(self, max_moves):
        board = chess.Board()

        n = random.randint(0, max_moves)
        for _ in range(n):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)

        return board

    # ----------------------------------------------------------
    #  Stockfish evaluation: returns a scalar value in [-1, 1]
    # ----------------------------------------------------------
    def evaluate(self, board):
        info = self.engine.analyse(board, limit=chess.engine.Limit(depth=self.stockfish_depth))

        score = info["score"].pov(board.turn)

        # Mate scores: squash into [-1, 1]
        if score.is_mate():
            return torch.tensor(1.0 if score.mate() > 0 else -1.0)

        # CP score: normalize (2000 cp → roughly winning)
        cp = score.score()
        return torch.tensor(max(-1.0, min(1.0, cp / 1000.0)), dtype=torch.float32)

    # ----------------------------------------------------------
    # AlphaZero-style encoding (simplified)
    # 119 planes:
    #   12 piece planes (6x2 colors)
    #   1 side-to-move plane
    #   4 castling rights
    #   1 repetition / move count (optional)
    #   + filler to reach 119 planes if desired
    # ----------------------------------------------------------
    def encode_board(self, board):
        planes = []

        # Piece type → plane index mapping
        piece_map = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

        # 12 planes: white pieces then black pieces
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                           chess.ROOK, chess.QUEEN, chess.KING]:

            # White piece plane
            p = np.zeros((8, 8), dtype=np.float32)
            for sq in board.pieces(piece_type, chess.WHITE):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                p[r, c] = 1.0
            planes.append(p)

            # Black piece plane
            p = np.zeros((8, 8), dtype=np.float32)
            for sq in board.pieces(piece_type, chess.BLACK):
                r = 7 - chess.square_rank(sq)
                c = chess.square_file(sq)
                p[r, c] = 1.0
            planes.append(p)

        # Side-to-move
        stm = np.ones((8, 8), dtype=np.float32) if board.turn == chess.WHITE else np.zeros((8, 8), dtype=np.float32)
        planes.append(stm)

        # Castling rights
        castling_planes = {
            chess.BB_H1: board.has_kingside_castling_rights(chess.WHITE),
            chess.BB_A1: board.has_queenside_castling_rights(chess.WHITE),
            chess.BB_H8: board.has_kingside_castling_rights(chess.BLACK),
            chess.BB_A8: board.has_queenside_castling_rights(chess.BLACK),
        }

        for right in castling_planes.values():
            plane = np.ones((8, 8), dtype=np.float32) if right else np.zeros((8, 8), dtype=np.float32)
            planes.append(plane)

        # If you want to pad to exactly 119 planes
        while len(planes) < 119:
            planes.append(np.zeros((8, 8), dtype=np.float32))

        stacked = np.stack(planes, axis=0)
        return torch.tensor(stacked, dtype=torch.float32)

    # ----------------------------------------------------------
    # PyTorch Dataset API
    # ----------------------------------------------------------
    def __len__(self):
        return self.n_positions

    def __getitem__(self, idx):
        return self.samples[idx]
