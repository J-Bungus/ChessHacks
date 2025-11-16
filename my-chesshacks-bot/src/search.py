import math
import torch
import chess
from train.src.utils import encode_board, MOVE_TO_IDX, encode_extra_state

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior  # P(s,a)

        self.children = {}  # move → MCTSNode

        self.N = 0  # visit count
        self.W = 0  # total simulation value
        self.Q = 0  # mean value

        self.is_expanded = False

    def expand(self, policy_probs, legal_moves):
        """Expand node with prior probabilities from policy head."""
        self.is_expanded = True
        for move in legal_moves:
            idx = MOVE_TO_IDX[str(move)]
            p = float(policy_probs[idx])
            next_board = self.board.copy()
            next_board.push(move)
            self.children[move] = MCTSNode(next_board, parent=self, prior=p)

class MCTS:
    def __init__(self, model, simulations=200, c_puct=1.4, device="cpu"):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device

    # --------------------------
    #   CORE MCTS LOGIC
    # --------------------------

    def run(self, root_board):
        root = MCTSNode(root_board)

        # Expand root immediately
        self._expand(root)

        for _ in range(self.simulations):
            node = root
            
            # 1. SELECTION
            while node.is_expanded and node.children:
                node = self._select(node)

            # 2. EXPANSION + EVALUATION
            value = self._expand(node)

            # 3. BACKUP
            self._backup(node, value)

        return root

    # --------------------------
    #   Selection
    # --------------------------
    def _select(self, node):
        """Select child that maximizes PUCT."""
        best_score = -float("inf")
        best_child = None

        sqrt_sum = math.sqrt(node.N + 1e-8)

        for move, child in node.children.items():
            uct = child.Q + self.c_puct * child.prior * sqrt_sum / (1 + child.N)
            if uct > best_score:
                best_score = uct
                best_child = child

        return best_child

    # --------------------------
    #   Expansion + Evaluation
    # --------------------------
    def _expand(self, node):
        """Expand node and evaluate with model (policy + value)."""
        if node.board.is_game_over():
            # Terminal value
            result = node.board.result()
            if result == "1-0": return 1
            if result == "0-1": return -1
            return 0

        # Convert board → tensor
        board_tensor = encode_board(node.board).unsqueeze(0).to(self.device)
        state = encode_extra_state(node.board).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.model(board_tensor, state)

        policy_probs = torch.softmax(policy_logits[0], dim=-1)

        legal_moves = list(node.board.generate_legal_moves())

        if not node.is_expanded:
            node.expand(policy_probs, legal_moves)

        return float(value.item())

    # --------------------------
    #   Backpropagate
    # --------------------------
    def _backup(self, node, value):
        """Propagate value up the tree."""
        cur = node
        while cur is not None:
            cur.N += 1
            cur.W += value
            cur.Q = cur.W / cur.N
            value = -value  # opponent's value is opposite
            cur = cur.parent

    # --------------------------
    #   Choosing the final move
    # --------------------------
    def choose_move(self, root):
        """Pick the move with highest visit count."""
        best_move = None
        best_N = -1

        for move, child in root.children.items():
            if child.N > best_N:
                best_N = child.N
                best_move = move

        return best_move
