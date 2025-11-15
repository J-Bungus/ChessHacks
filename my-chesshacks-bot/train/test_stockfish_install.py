import chess.engine

STOCKFISH_PATH = r"C:\Users\mike-\Downloads\applications\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"  # path to engine

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
print("Engine OK!")
engine.quit()
