import torch
import chess

BOARD_VEC_SIZE = 13  # 6 piece types per color + 1 for empty
NUM_PIECE_TYPES = 6
PIECE_TYPES = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def encode_board(board: chess.Board):
    """
    Encode board as integer indices (0-12) for nn.Embedding.
    0-5: white pieces, 6-11: black pieces, 12: empty square
    Output shape: (64,)
    """
    enc = torch.zeros(64, dtype=torch.long)

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece is None:
            enc[square] = 12  # empty square
        else:
            base_idx = PIECE_TYPES[piece.piece_type]
            if piece.color == chess.BLACK:
                base_idx += NUM_PIECE_TYPES
            enc[square] = base_idx

    return enc  # (64,)

def encode_extra_state(board: chess.Board):
    """
    Encode extra state as integer indices for embedding (if desired, could use same embedding approach)
    Shape: (5,)
    """
    return torch.tensor([
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
        int(board.turn == chess.WHITE)
    ], dtype=torch.long)  # (5,)


# ---- Move encoding utilities: 4672 legal UCI moves ----
ALL_SQUARES = [f"{f}{r}" for f in "abcdefgh" for r in "12345678"]
ALL_MOVES = []

for s1 in ALL_SQUARES:
    for s2 in ALL_SQUARES:
        if s1 == s2:
            continue

        ALL_MOVES.append(s1 + s2)  # quiet + capture

        # promotions
        if s1[1] == "7" and s2[1] == "8":
            for promo in ["q", "r", "b", "n"]:
                ALL_MOVES.append(s1 + s2 + promo)
        elif s1[1] == "2" and s2[1] == "1":
            for promo in ["q", "r", "b", "n"]:
                ALL_MOVES.append(s1 + s2 + promo)

MOVE_TO_IDX = {m: i for i, m in enumerate(ALL_MOVES)}
NUM_MOVES = len(ALL_MOVES)
