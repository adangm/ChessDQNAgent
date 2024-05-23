import chess
import chess.engine
import os

class StockfishAgent:
    def __init__(self, stockfish_path):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    def generate_move(self, board):
        result = self.engine.play(board, chess.engine.Limit(time=0.1))
        return result.move.uci()

    def close(self):
        self.engine.quit()
