import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import chess
import chess.engine
import numpy as np
import random
from multiprocessing import Process, Queue
from lightning import LightningDataModule
from collections import deque
import time
from tqdm import tqdm
from utils import encode_board, encode_extra_state, MOVE_TO_IDX
import os


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
                 history_length=1, 
                 train=False):

        self.num_games = num_games
        self.max_moves = max_moves
        self.stockfish_path = stockfish_path
        self.movetime = movetime
        self.history_length = history_length

        self.samples = []
        os.makedirs("./cache", exist_ok=True)
        self.cache_path = "./cache/train_data.pt" if train else "./cache/val_data.pt"

        if os.path.exists(self.cache_path):
            print(f"Loading dataset from cache: {self.cache_path}")
            self.samples = torch.load(self.cache_path)
            print(f"Loaded {len(self.samples)} samples.")
        else:
            print(f"No cache found. Generating {self.num_games} games...")
            self.samples = self._generate_games(self.num_games)
            torch.save(self.samples, self.cache_path)
            print(f"Saved {len(self.samples)} samples to cache: {self.cache_path}")

    def _generate_games(self, num_games: int,):
        """Generates a number of self-play games and returns a list of samples."""
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        all_samples = []

        for _ in tqdm(range(num_games), desc="Generating games"):
            board = chess.Board()
            game_samples = []

            for _ in range(self.max_moves):
                if board.is_game_over():
                    break

                # Get top 20 moves from Stockfish
                info_list = engine.analyse(board, chess.engine.Limit(time=self.movetime), multipv=10)

                if not info_list:
                    break

                # Collect moves and evaluations
                moves, evals = [], []
                for info in info_list:
                    move = info["pv"][0]
                    eval_cp = info["score"].white().score(mate_score=100000)
                    if eval_cp is None or move.uci() not in MOVE_TO_IDX:
                        continue
                    moves.append(move)
                    evals.append(eval_cp)

                if not moves:
                    break

                # Sample a move according to probabilities for actual play
                stm_sign = 1 if board.turn == chess.WHITE else -1
                stm_evals = np.array([stm_sign * e for e in evals], dtype=np.float32)
                stm_evals = np.clip(stm_evals, -100000, 100000)
                probs = np.exp(stm_evals / 200.0)
                probs /= probs.sum()

                if np.isnan(probs).any():
                    played_move = moves[0]
                else:
                    played_move = np.random.choice(moves, p=probs)

                board.push(played_move)

                # Determine best move for dataset target
                best_idx = int(np.argmax([stm_sign * e for e in evals]))
                best_move = moves[best_idx]
                best_eval = evals[best_idx]

                # Encode board + state for dataset (policy_target = best move)
                board_tokens = encode_board(board)
                state = encode_extra_state(board)
                policy_idx = MOVE_TO_IDX[best_move.uci()]
                v = torch.tensor(eval_cp_to_target(best_eval), dtype=torch.float32)

                game_samples.append((board_tokens, state, policy_idx, v))


            all_samples.extend(game_samples)

        engine.quit()
        return all_samples


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def resample(self, fraction: float = 0.1):
        """Replace a fraction of the dataset with newly generated games."""
        # num_replace = max(1, int(len(self.samples) * fraction))
        num_replace = 30
        new_samples = self._generate_games(num_replace)

        replace_indices = random.sample(range(len(self.samples)), len(new_samples))
        for idx, new_sample in zip(replace_indices, new_samples):
            self.samples[idx] = new_sample

class SelfPlayDataModule(LightningDataModule):
    def __init__(self, stockfish_path, batch_size=32, num_workers=0, num_train_games=10, num_val_games=10):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stockfish_path = stockfish_path
        self.num_train_games=num_train_games
        self.num_val_games = num_val_games

    def train_dataloader(self):
        dataset = SelfPlayDataset(
            stockfish_path=self.stockfish_path,
            num_games=self.num_train_games,
            train=True
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=0,   # <--- IMPORTANT
            pin_memory=True
        )

    def val_dataloader(self):
        # Small validation dataset (fixed 10 games)
        val_dataset = SelfPlayDataset(
            stockfish_path=self.stockfish_path,
            num_games=self.num_val_games,
            train=False
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

