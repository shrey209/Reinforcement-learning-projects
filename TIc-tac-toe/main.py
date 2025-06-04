import random
import numpy as np
import json

class TicTacToe:
    def __init__(self, players):
        self.players = players
        self.board = [" " for _ in range(9)]
        self.gameOver = False
        self.winner = ""

    def displayBoard(self):
        print("Current board:")
        print(" ", self.board[0], "|", self.board[1], "|", self.board[2])
        print("---|---|---")
        print(" ", self.board[3], "|", self.board[4], "|", self.board[5])
        print("---|---|---")
        print(" ", self.board[6], "|", self.board[7], "|", self.board[8])

    def playGame(self):
        while not self.gameOver:
            move = self.players[0].makeMove(self.board)
            self.board[move] = "X"
            self.checkWin()
            if self.gameOver:
                break
            move = self.players[1].makeMove(self.board)
            self.board[move] = "O"
            self.checkWin()

    def checkWin(self):
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for combo in winning_combinations:
            a, b, c = combo
            if self.board[a] != " " and self.board[a] == self.board[b] == self.board[c]:
                self.winner = 0 if self.board[a] == "X" else 1
                self.gameOver = True
                return
        if " " not in self.board:
            self.gameOver = True
            self.winner = 2  # Tie

    def reset(self):
        self.board = [" " for _ in range(9)]
        self.gameOver = False
        self.winner = ""


class Player:
    def __init__(self, name, strategy, epsilon):
        self.name = name
        self.strategy = strategy
        self.epsilon = epsilon
        self.q_table = {}
        self.states = []

    def makeMove(self, board):
        validMoves = [i for i in range(len(board)) if board[i] == " "]
        if self.strategy == "human":
            print("Your valid moves are:", validMoves)
            move = int(input("Where would you like to move? "))
            return move
        elif self.strategy == "random":
            return random.choice(validMoves)
        elif self.strategy == "AI":
            state = tuple(board)
            if state not in self.q_table:
                self.q_table[state] = [0 for _ in validMoves]
            if random.random() < self.epsilon:
                move_index = random.randint(0, len(validMoves) - 1)
            else:
                move_index = np.argmax(self.q_table[state])
            move = validMoves[move_index]
            self.states.append((state, move_index))
            return move

    def updateQTable(self, reward):
        alpha = 0.1
        gamma = 0.9
        for state, action_index in self.states:
            max_future_q = max(self.q_table[state])
            self.q_table[state][action_index] += alpha * (reward + gamma * max_future_q - self.q_table[state][action_index])
        self.states.clear()

    def save_q_table(self, filename="q_table.jsonl"):
        with open(filename, "w") as f:
            for state, values in self.q_table.items():
                entry = {"state": list(state), "q_values": values}
                f.write(json.dumps(entry) + "\n")


def main():
    player1 = Player("Player1", "AI", 0.99)
    player2 = Player("Player2", "AI", 0.99)
    players = [player1, player2]
    game = TicTacToe(players)

    for _ in range(1000000):
        game.playGame()
        if game.winner == 0:
            player1.updateQTable(1)
            player2.updateQTable(-1)
        elif game.winner == 1:
            player1.updateQTable(-1)
            player2.updateQTable(1)
        else:
            player1.updateQTable(0.5)
            player2.updateQTable(0.5)
        game.reset()

    player1.epsilon = 0
    game2 = TicTacToe(players)
    wins = losses = ties = 0

    for _ in range(100000):
        game2.playGame()
        if game2.winner == 0:
            wins += 1
        elif game2.winner == 1:
            losses += 1
        else:
            ties += 1
        game2.reset()

    print("Wins:", wins, "Losses:", losses, "Ties:", ties)
    print("Tic-tac-toe game completed!")

    # Save Q-tables to files
    player1.save_q_table("player1_q_table.jsonl")
    player2.save_q_table("player2_q_table.jsonl")


if __name__ == "__main__":
    main()
