import torch
import chess
import chess.engine
import utils
from torch.utils.data import IterableDataset

class OnlineSelfPlayDataset(IterableDataset):
    """
    Online generator:
      - opens a Stockfish engine in __iter__
      - plays self-play games
      - yields (x, policy_idx, value) tuples on the fly

    positions_per_epoch: approx # of positions to yield per epoch
    """

    def __init__(self,
                 n_positions=5000,
                 max_moves=200,
                 stockfish_path="/usr/bin/stockfish",
                 movetime=0.01):
        super().__init__()
        self.positions_per_epoch = n_positions
        self.max_moves = max_moves
        self.stockfish_path = stockfish_path
        self.movetime = movetime

    def _sample_generator(self):
        # This runs inside the worker process
        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        generated = 0

        try:
            while generated < self.positions_per_epoch:
                # Play one self-play game
                positions, moves, evals_cp = utils.play_self_play_game(
                    engine,
                    max_moves=self.max_moves,
                    movetime=self.movetime
                )

                if len(moves) == 0:
                    continue
                
                rep_flags_all = utils.compute_repetition_flags(positions)

                for i, (board, move, eval_cp) in enumerate(zip(positions, moves, evals_cp)):
                    # Build 8-position history
                    history_boards = utils.get_history_stack(positions, i)
                    rep_flags = rep_flags_all[i]

                    # Encode to (64, 112)
                    x = utils.encode_history_to_tokens(history_boards, rep_flags)  # torch.Tensor

                    # Policy target
                    uci = move.uci()
                    idx = utils.MOVE_TO_IDX.get(uci, None)
                    if idx is None:
                        continue  # skip rare moves that fall outside your encoding
                    policy_idx = torch.tensor(idx, dtype=torch.long)

                    # Value target: eval (cp) from side-to-move POV → [-1,1]
                    v = utils.eval_cp_to_target(eval_cp)
                    v = torch.tensor(v, dtype=torch.float32)

                    yield x, policy_idx, v
                    generated += 1

                    if generated >= self.positions_per_epoch:
                        break
        finally:
            engine.quit()

    def __iter__(self):
        # DataLoader will call this per worker
        return self._sample_generator()
