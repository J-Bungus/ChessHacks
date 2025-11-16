import chess
import chess.pgn
import chess.engine
import pandas as pd
import itertools
from tqdm import tqdm

# --------------------------------------------------------------
# Stockfish (or any UCI engine) Initialization
# --------------------------------------------------------------
STOCKFISH_PATH = r"C:\Users\mike-\Downloads\applications\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"  # path to engine
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def evaluate_position(board: chess.Board, depth=12):
    """
    Uses python-chess.engine to evaluate a position.
    Returns score in pawns (float).
    """
    info = engine.analyse(board, chess.engine.Limit(depth=depth))

    score = info["score"].pov(chess.WHITE)

    if score.is_mate():
        # Large positive or negative value
        return 1000.0 if score.mate() > 0 else -1000.0
    else:
        return score.score() / 100.0  # convert centipawns → pawns

def load_positions_from_pgn(pgn_path, max_positions=500):
    """
    Yield up to max_positions unique positions from games in a PGN.
    """
    positions = 0

    with open(pgn_path) as f:
        while positions < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            for move in game.mainline_moves():
                yield board.copy()
                positions += 1
                if positions >= max_positions:
                    return
                board.push(move)

def generate_sibling_pairs(board: chess.Board):
    """
    For a given board, generate all pairs of legal child positions
    labeled according to Stockfish's ranking.
    """
    legal = list(board.legal_moves)
    if len(legal) < 2:
        return

    children = []
    for mv in legal:
        child = board.copy()
        child.push(mv)
        score = evaluate_position(child)
        children.append((child, score))

    # All unique pairs of siblings
    for (b1, s1), (b2, s2) in itertools.combinations(children, 2):
        if s1 == s2:
            continue
        label = 1 if s1 > s2 else -1
        yield b1.fen(), b2.fen(), label

def build_csv_dataset(pgn_path, csv_path, max_positions=500):
    """
    Extract sibling pairs from up to max_positions PGN positions
    and save them to a CSV file.
    """
    rows = []

    for board in tqdm(load_positions_from_pgn(pgn_path, max_positions=max_positions), total=max_positions):
        for fen_a, fen_b, label in generate_sibling_pairs(board):
            rows.append({
                "fen_a": fen_a,
                "fen_b": fen_b,
                "label": label
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    print(f"Saved {len(df)} samples → {csv_path}")
    return df

df = build_csv_dataset(
    pgn_path="games.pgn",
    csv_path="pairs.csv",
    max_positions=2000
)

engine.quit()
