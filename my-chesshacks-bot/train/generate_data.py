import chess
import chess.pgn
import chess.engine
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
import os
import time

# ---------------------------------------
# CONFIG
# ---------------------------------------
PGN_FILE = "games.pgn"      # your PGN file
OUTPUT_CSV = "stockfish_positions.csv"
STOCKFISH_PATH = r"C:\Users\mike-\Downloads\applications\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"  # path to engine
EVAL_DEPTH = 12
NUM_WORKERS = 4             # Number of parallel Stockfish engines
MAX_POSITIONS = 50000
SAMPLE_EVERY = 1            # Evaluate every Nth ply
CHUNK_SIZE = 200            # How often to flush results to disk

# ---------------------------------------------------------
# Worker: runs in each process
# ---------------------------------------------------------
def worker_process(task_queue, result_queue, worker_id):

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    while True:
        item = task_queue.get()

        if item == "STOP":
            break

        fen = item
        board = chess.Board(fen)

        # Evaluate position
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=EVAL_DEPTH))
            score = info["score"].pov(chess.WHITE)

            if score.is_mate():
                eval_score = 10 if score.mate() > 0 else -10
            else:
                eval_score = score.score() / 100.0  # convert cp to pawns

        except Exception as e:
            eval_score = None

        # Send result back
        result_queue.put((fen, eval_score))

    engine.quit()


# ---------------------------------------------------------
# CSV Writer: runs in main process
# ---------------------------------------------------------
def writer_process(result_queue, total_expected):

    buffer = []
    processed = 0

    # Create CSV with header if missing
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "w") as f:
            f.write("fen,score\n")

    pbar = tqdm(total=total_expected, desc="Evaluating positions")

    while processed < total_expected:
        fen, score = result_queue.get()
        processed += 1
        pbar.update(1)

        buffer.append((fen, score))

        # Flush buffer periodically
        if len(buffer) >= CHUNK_SIZE:
            df = pd.DataFrame(buffer, columns=["fen", "score"])
            df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)
            buffer = []

    # Final flush
    if buffer:
        df = pd.DataFrame(buffer, columns=["fen", "score"])
        df.to_csv(OUTPUT_CSV, mode="a", header=False, index=False)

    pbar.close()
    print("\nFinished writing results.")


# ---------------------------------------------------------
# Main PGN reader and process launcher
# ---------------------------------------------------------
def generate_dataset_parallel():
    print("Scanning PGN for positions...")

    # First pass: extract all FENs
    fens = []
    with open(PGN_FILE, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)

                if ply % SAMPLE_EVERY == 0:
                    fens.append(board.fen())

                if len(fens) >= MAX_POSITIONS:
                    break

            if len(fens) >= MAX_POSITIONS:
                break

    total_positions = len(fens)
    print(f"Collected {total_positions} positions.")

    # Create multiprocessing queues
    task_queue = mp.Queue(maxsize=2000)
    result_queue = mp.Queue(maxsize=2000)

    # Start workers
    workers = []
    for i in range(NUM_WORKERS):
        p = mp.Process(target=worker_process, args=(task_queue, result_queue, i))
        p.start()
        workers.append(p)

    # Start writer in the main process
    writer = mp.Process(target=writer_process, args=(result_queue, total_positions))
    writer.start()

    # Feed tasks
    for fen in fens:
        task_queue.put(fen)

    # Send STOP signals
    for _ in workers:
        task_queue.put("STOP")

    # Wait for workers to finish
    for p in workers:
        p.join()

    # Wait for writer to finish
    writer.join()

    print("Dataset generation complete!")


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn")  # Required on Windows
    generate_dataset_parallel()