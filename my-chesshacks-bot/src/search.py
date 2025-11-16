import math
import torch
import chess
from train.src.utils import compute_repetition_flags, encode_history_to_tokens, MOVE_TO_IDX, encode_extra_state, get_history_stack

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
        self.positions = []

    # --------------------------
    #   CORE MCTS LOGIC
    # --------------------------

    def run(self, root_board):
        root = MCTSNode(root_board)
        self.positions.append(root_board)

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
    
    def _build_history_tokens(self, node: MCTSNode) -> torch.Tensor:
        """
        Reconstruct the path from root to `node`, then use your utils:
          - compute_repetition_flags
          - get_history_stack
          - encode_history_to_tokens

        This works for any HISTORY_LEN configured in your utils.
        """
        # Gather boards from root → node
        path_boards = []
        cur = node
        while cur is not None:
            path_boards.append(cur.board)
            cur = cur.parent
        path_boards.reverse()  # now oldest..newest (root first, node last)

        # Compute repetition flags for the whole path
        rep_flags_all = compute_repetition_flags(path_boards)

        # Index of current board in that list
        idx = len(path_boards) - 1

        # HISTORY_LEN is handled internally by get_history_stack
        history_boards = get_history_stack(path_boards, idx)
        rep_flags = rep_flags_all[idx]

        # encode_history_to_tokens should return [64, feature_dim] as a torch.Tensor
        tokens = encode_history_to_tokens(history_boards, rep_flags)  # [64, feature_dim]
        return tokens



    # --------------------------
    #   Expansion + Evaluation
    # --------------------------
    def _expand(self, node):
        """Expand node and evaluate with model (policy + value)."""
        if node.board.is_game_over():
            result = node.board.result()
            # value from *current player’s* POV
            if result == "1-0":  # white won
                return 1 if node.board.turn == chess.WHITE else -1
            if result == "0-1":  # black won
                return 1 if node.board.turn == chess.BLACK else -1
            return 0

        # Convert board → tensor
        
        board_feats = self._build_history_tokens(node)              # [64, feature_dim]
        board_tensor = board_feats.unsqueeze(0).to(self.device)   
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)
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
