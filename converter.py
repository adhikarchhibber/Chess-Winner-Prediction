import pandas as pd
import chess
import chess.pgn
import io
import chess
import chess.pgn
import chess.engine
import io
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
#we are going to examine games.csv and make our evaluations and tenor conversions so the model can be ready to load them in

df = pd.read_csv("games.csv")
df.dropna()
#convert white_elo and black_elo to both types int
df['white_elo'] = df['white_elo'].astype(int)
df['black_elo'] = pd.to_numeric(df['black_elo'], errors='coerce')
df = df.dropna(subset=['black_elo'])
df['black_elo'] = df['black_elo'].astype(int)
#result is in 1-0, needs to be 1 for white and 0 for black
df['result'] = df['result'].apply(lambda x: 1 if x == '1-0' else 0)
df['average_rating'] = (df['white_elo'] + df['black_elo']) / 2
df = df.sort_values(by='average_rating', ascending=False)[:100000]

import chess
import chess.pgn
import chess.engine
import io
import torch
from tqdm import tqdm
import pandas as pd
import pickle

# Constants
stockfish_path = "/usr/local/bin/stockfish"
MAX_EVAL = 1000

def normalize_eval(score):
    if score is None:
        return 0.0
    clamped = max(-MAX_EVAL, min(MAX_EVAL, score))
    return clamped / MAX_EVAL

def fen_to_tensor(fen):
    piece_map = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    board_tensor = torch.zeros(12, 8, 8, dtype = torch.float32)
    board = chess.Board(fen)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_idx = piece_map[piece.symbol()]
            rank = 7 - (square // 8)
            file = square % 8
            board_tensor[piece_idx, rank, file] = 1.0

    return board_tensor

def process_game(moves_str, engine):
    try:
        pgn = io.StringIO("1. " + moves_str)
        game = chess.pgn.read_game(pgn)
        board = game.board()

        fens = [board.fen()]
        evals = [0.0]

        for i, move in enumerate(game.mainline_moves()):
            if i >= 16:
                break
            board.push(move)
            fens.append(board.fen())
            info = engine.analyse(board, chess.engine.Limit(depth = 8), multipv = 1)
            #print(info)
            score = info[0]["score"].white().score(mate_score = 10000)
            #score = 0
            #print('\ngot_info\n')
            evals.append(normalize_eval(score))

        fen_tensors = torch.stack([fen_to_tensor(fen) for fen in fens])
        eval_tensor = torch.tensor(evals, dtype = torch.float32)

        return fen_tensors, eval_tensor

    except Exception as e:
        print(f"Error processing game: {e}")
        return None, None

def add_tensors_to_df(df):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 32})
    game_tensors = []
    eval_tensors = []

    for moves in tqdm(df['moves'], desc="Processing games"):
        fen_t, eval_t = process_game(moves, engine)
        if fen_t is not None and eval_t is not None:
            game_tensors.append(fen_t)
            eval_tensors.append(eval_t)
        else:
            game_tensors.append(None)
            eval_tensors.append(None)

    engine.quit()

    df['game_tensor'] = game_tensors
    df['eval_tensor'] = eval_tensors

    df_clean = df.dropna(subset=['game_tensor', 'eval_tensor']).reset_index(drop=True)
    return df_clean
df_clean = add_tensors_to_df(df)
df_clean.to_pickle("df_clean.pkl")

#df_clean['game_tensor'] = df_clean['game_tensor'].apply(lambda x: x.tolist())
#df_clean['eval_tensor'] = df_clean['eval_tensor'].apply(lambda x: x.tolist())
#df_clean.to_csv("processed_chess_games.csv", index=False)
