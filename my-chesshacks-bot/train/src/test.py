from data import SelfPlayDataset

data = SelfPlayDataset(1, 50, '/usr/games/stockfish', 0.05)
print(data.samples[-1])