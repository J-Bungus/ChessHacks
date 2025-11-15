import torch
from torch.utils.data import Dataset
import chess
import chess.engine
import numpy as np
import random
from utils import encode_board, MOVE_TO_IDX, NUM_MOVES

class ChessPolicyValueDataset(Dataset):
    """
    Pretraining dataset:
    Outputs:
       x: (18,8,8) board planes
       policy_target: integer index or soft distribution
       value_target: scalar in [-1,1]
    """

    def __init__(self,
                 n_positions=5000,
                 max_random_moves=20,
                 stockfish_path="/usr/bin/stockfish",
                 stockfish_depth=12,
                 multipv=1):  # multipv>1 gives soft distribution
        
        self.n_positions = n_positions
        self.max_random_moves = max_random_moves
        self.depth = stockfish_depth
        self.multipv = multipv

        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        
        self.samples = []

        for _ in range(n_positions):
            board = self.random_position(max_random_moves)

            result = self.query_stockfish(board)
            if result is None:
                continue

            policy_target, value_target = result

            x = encode_board(board)
            self.samples.append((x, policy_target, value_target))

        self.engine.quit()

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
    # Stockfish query: returns (policy_target, value_target)
    # ----------------------------------------------------------
    def query_stockfish(self, board):
        info = self.engine.analyse(
            board,
            limit=chess.engine.Limit(depth=self.depth),
            multipv=self.multipv
        )

        if self.multipv == 1:
            info = [info]

        # ---------- Value target ----------
        # Use the PV #1 evaluation
        score = info[0]["score"].white().score(mate_score=100000)

        if score is None:
            return None

        # convert cp → [-1,1]
        # typical cp / 400 works well
        value_target = np.tanh(score / 400.0)

        # ---------- Policy target ----------
        if self.multipv == 1:
            # Just the best move index
            best_move = info[0]["pv"][0]
            move_idx = MOVE_TO_IDX.get(best_move.uci(), None)
            if move_idx is None:
                return None
            policy_target = move_idx  # integer index
        else:
            # Create a (4672,) distribution
            pi = np.zeros(NUM_MOVES, np.float32)

            scores = []
            moves = []
            for line in info:
                if "pv" not in line:
                    continue  # skip empty PV entries
                mv = line["pv"][0]
                mv_idx = MOVE_TO_IDX.get(mv.uci(), None)
                assert mv_idx is not None
                eval_cp = line["score"].white().score(mate_score=100000)
                scores.append(eval_cp)
                moves.append(mv_idx)

            if len(scores) == 0:
                return None

            # softmax on scores
            scores = torch.tensor(scores).to(dtype=torch.float32)
            probs = torch.nn.functional.softmax(scores, dim=0).to(dtype=torch.float32)

            for mv_idx, p in zip(moves, probs):
                pi[mv_idx] = p

            policy_target = torch.tensor(pi, dtype=torch.float32)

        return policy_target, torch.tensor(value_target).to(dtype=torch.float32)

    # ----------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
