import chess
from chess_env import ChessEnv
from dqn_agent import DQNAgent
from stockfish_agent import StockfishAgent

def evaluate_model(dqn_agent, stockfish_agent, episodes=100):
    env = ChessEnv()
    dqn_wins = 0
    stockfish_wins = 0
    draws = 0

    for episode in range(episodes):
        # Reiniciar el tablero de ajedrez y el entorno
        board = chess.Board()
        state = env.reset()
        done = False
        step_count = 0  # Contador de pasos dentro de un episodio
        print(f"Starting episode {episode + 1}/{episodes}")
        while not done:
            step_count += 1
            # Turno del agente DQN
            state = env._get_obs()
            state = state.reshape((1, *state.shape))
            action = dqn_agent.act(state)
            next_state, reward, done, _ = env.step(action)
            if done:
                if reward == 1:
                    dqn_wins += 1
                elif reward == -1:
                    stockfish_wins += 1
                else:
                    draws += 1
                break
            
            # Turno del agente Stockfish
            stockfish_move = stockfish_agent.generate_move(board)
            board.push_uci(stockfish_move)
            if board.is_game_over():
                if board.result() == "1-0":
                    dqn_wins += 1
                elif board.result() == "0-1":
                    stockfish_wins += 1
                else:
                    draws += 1
                break

            # Actualizar el estado del entorno después del movimiento de Stockfish
            env.board = board
            next_state = env._get_obs()

            print(f"Episode: {episode + 1}/{episodes}, Step: {step_count}, DQN Action: {action}, Stockfish Move: {stockfish_move}")

    print(f"\nEvaluation completed.")
    print(f"DQN Wins: {dqn_wins}")
    print(f"Stockfish Wins: {stockfish_wins}")
    print(f"Draws: {draws}")

if __name__ == "__main__":
    dqn_agent = DQNAgent(state_size=(8, 8, 12), action_size=4672)
    dqn_agent.load("chess_dqn_model_dataset_trained")

    # Reemplaza "path/to/stockfish" con la ruta al ejecutable de Stockfish en tu sistema
    stockfish_agent = StockfishAgent(stockfish_path=r"C:\Users\PC\Desktop\stockfish\stockfish-windows-x86-64-avx2.exe")

    evaluate_model(dqn_agent, stockfish_agent)

    # Cerrar el motor de Stockfish después de la evaluación
    stockfish_agent.close()

