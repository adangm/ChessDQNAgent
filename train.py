import time
import json
import pandas as pd
import chess
import chess.pgn
from chess_env import ChessEnv
from dqn_agent import DQNAgent
import numpy as np
import os
import pickle

# Cargar el dataset
dataset_path = "games.csv"
print("Cargando el dataset...")
games_df = pd.read_csv(dataset_path)
print("Dataset cargado.")

# Filtrar columnas relevantes
games_df = games_df[['id', 'white_id', 'white_rating', 'black_id', 'black_rating', 'moves', 'opening_name', 'winner']]

# Función para convertir la secuencia de movimientos en un objeto de juego de ajedrez
def moves_to_game(moves):
    game = chess.pgn.Game()
    node = game
    board = chess.Board()
    for move in moves.split():
        try:
            uci_move = board.push_san(move).uci()
            node = node.add_variation(chess.Move.from_uci(uci_move))
        except Exception as e:
            print(f"Invalid move: {move} -> {e}")
            continue  # Saltar movimientos inválidos y continuar con el siguiente
    return game

# Convertir los movimientos a juegos
print("Convirtiendo movimientos a juegos...")
games_df['game'] = games_df['moves'].apply(moves_to_game)
print("Movimientos convertidos a juegos.")

# Función para convertir un tablero de ajedrez a una representación de estado
def board_to_state(board):
    state = np.zeros((8, 8, 12), dtype=int)
    pieces = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for piece, index in pieces.items():
        for square in board.pieces(piece, chess.WHITE):
            state[chess.square_rank(square), chess.square_file(square), index] = 1
        for square in board.pieces(piece, chess.BLACK):
            state[chess.square_rank(square), chess.square_file(square), index + 6] = 1
    return state

# Generar experiencias de entrenamiento con límite
print("Generando experiencias de entrenamiento...")
def generate_experiences(games_df, max_experiences=50000):
    experiences = []
    for idx, row in games_df.iterrows():
        if len(experiences) >= max_experiences:
            break
        if idx % 100 == 0:
            print(f"Procesando juego {idx + 1}/{len(games_df)}...")
        game = row['game']
        board = game.board()
        for move in game.mainline_moves():
            state = board_to_state(board)
            action = move.uci()
            board.push(move)
            next_state = board_to_state(board)
            reward = 1 if row['winner'] == 'white' else -1 if row['winner'] == 'black' else 0
            done = board.is_game_over()
            experiences.append((state, action, reward, next_state, done))
            if done or len(experiences) >= max_experiences:
                break
    return experiences

experiences = generate_experiences(games_df, max_experiences=50000)
print(f"Experiencias de entrenamiento generadas: {len(experiences)}")

# Inicializar el agente DQN
state_size = (8, 8, 12)
action_size = 4672  # Número máximo de movimientos posibles en ajedrez
agent = DQNAgent(state_size, action_size)
env = ChessEnv()

# Cargar el modelo guardado y el estado del entrenamiento si existen
start_episode = 0
model_filename = "chess_dqn_model_latest.keras"
training_state_filename = "training_state.json"
experiences_filename = "experiences.pkl"

if os.path.exists(model_filename):
    agent.load(model_filename)
    print(f"Modelo cargado desde '{model_filename}'")
    if os.path.exists(training_state_filename):
        with open(training_state_filename, "r") as file:
            training_state = json.load(file)
            start_episode = training_state.get("episode", 0)
            print(f"Continuando el entrenamiento desde el episodio {start_episode + 1}")
    if os.path.exists(experiences_filename):
        with open(experiences_filename, "rb") as file:
            agent.memory = pickle.load(file)
            print(f"Memoria de experiencias cargada con {len(agent.memory)} experiencias")

start_time = time.time()  # Inicio del tiempo de entrenamiento

# Función para convertir una acción en UCI a un índice
def uci_to_index(move):
    legal_moves = list(env.board.legal_moves)
    return legal_moves.index(chess.Move.from_uci(move))

# Entrenar el agente DQN con las experiencias generadas del dataset
print("Entrenando el agente DQN con las experiencias del dataset...")
batch_size = 64  # Tamaño del batch para el replay, incrementado para acelerar el proceso
for idx, (state, action, reward, next_state, done) in enumerate(experiences):
    if idx % 1000 == 0:
        print(f"Procesando experiencia {idx + 1}/{len(experiences)}...")
    try:
        action_index = uci_to_index(action)
        agent.remember(state, action_index, reward, next_state, done)
        if len(agent.memory) > batch_size:
            print(f"Training on batch... {len(agent.memory)} experiencias en memoria")
            agent.replay(batch_size)
    except ValueError:
        print(f"Invalid move {action} at experience {idx + 1}")

# Guardar el modelo entrenado
agent.save("chess_dqn_model_dataset_trained.keras")
print("Modelo entrenado guardado.")

print("Entrenamiento inicial completado. Comenzando episodios de entrenamiento...")

# Entrenar el agente DQN por episodios
episodes = 100  # Número total de episodios
batch_size = 64

for e in range(start_episode, episodes):
    state = env.reset()
    state = np.reshape(state, [1, state.shape[0], state.shape[1], state.shape[2]])
    total_reward = 0
    for time_step in range(200):  # Limitar los pasos por episodio
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, next_state.shape[0], next_state.shape[1], next_state.shape[2]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (e + 1)) * episodes
            remaining_time = estimated_total_time - elapsed_time
            print(f"Episode: {e+1}/{episodes}, Score: {time_step}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            print(f"Elapsed Time: {elapsed_time:.2f}s, Estimated Total Time: {estimated_total_time:.2f}s, Remaining Time: {remaining_time:.2f}s")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # Información sobre cada paso
        print(f"Episode: {e+1}/{episodes}, Step: {time_step}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}")

    # Guardar el modelo al final de cada episodio
    agent.save("chess_dqn_model_latest")
    if e % 10 == 0:  # Guarda el modelo cada 10 episodios con un nombre específico
        agent.save(f"chess_dqn_model_{e}")

    # Guardar el estado del entrenamiento
    with open(training_state_filename, "w") as file:
        json.dump({"episode": e + 1}, file)

    # Guardar la memoria de experiencias
    with open(experiences_filename, "wb") as file:
        pickle.dump(agent.memory, file)
