import chess
import chess.engine
import gym
from gym import spaces
import numpy as np

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.action_space = spaces.Discrete(4672)  # Número máximo de movimientos posibles en ajedrez
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=int)

    def reset(self):
        self.board.reset()
        return self._get_obs()

    def step(self, action):
        if isinstance(action, np.ndarray) and action.size == 1:
            action = action.item()
        if isinstance(action, (np.ndarray, list)):
            action = action[0]
        move = self._decode_action(action)
        if move in self.board.legal_moves:
            self.board.push(move)
            done = self.board.is_game_over()
            reward = self._get_reward()
            return self._get_obs(), reward, done, {}
        else:
            return self._get_obs(), -1, False, {}

    def _get_obs(self):
        # Representa el tablero como una matriz 8x8x12
        board_state = np.zeros((8, 8, 12), dtype=int)
        pieces = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        for piece, index in pieces.items():
            for square in self.board.pieces(piece, chess.WHITE):
                board_state[chess.square_rank(square), chess.square_file(square), index] = 1
            for square in self.board.pieces(piece, chess.BLACK):
                board_state[chess.square_rank(square), chess.square_file(square), index + 6] = 1
        return board_state

    def _decode_action(self, action):
        # Decodifica la acción en un movimiento de ajedrez
        legal_moves = list(self.board.legal_moves)
        if isinstance(action, np.ndarray) and action.size == 1:
            action = action.item()
        if isinstance(action, (np.ndarray, list)):
            action = action[0]
        if action < len(legal_moves):
            move = legal_moves[action]
            print(f"Decoding action: {action}, Move: {move}, Legal Moves: {legal_moves}")  # Agregar impresión detallada
            return move
        else:
            return chess.Move.null()


    def _get_reward(self):
        if self.board.is_checkmate():
            return 1
        elif self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves() or self.board.is_fivefold_repetition():
            return 0.5
        else:
            return 0
