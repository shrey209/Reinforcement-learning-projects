import random 
import numpy as np

class TicTacToe:
    def __init__(self,players):
        self.players=players
        self.board=[" " for x in range(9)]
        self.gameOver = False
        self.winner = ""
    
    def displayBoard(self):
        print("Current board:")
        print(" ", self.board[0], "|", self.board[1], "|", self.board[2])
        print("---|---|---")
        print(" ", self.board[3], "|", self.board[4], "|", self.board[5])
        print("---|---|---")
        print(" ", self.board[6], "|", self.board[7], "|", self.board[8])

    def play(self):
        while(True):
            move = self.players[0].makeMove(self.board)
            self.board[move] = "X"
            self.checkForWin()
            if(self.gameOver):
                break
            move = self.players[1].makeMove(self.board)
            self.board[move] = "O"
            self.checkForWin()
            if(self.gameOver):
                break
    
    def checkWin(self):
        winning_combinations = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6]
        ]
        for combination in winning_combinations:
           if  self.board[combination[0]]!=" " and self.board[combination[0]] ==self.board[combination[1]] == self.board[combination[2]] :
                if(self.board[i] == "X"):
                    self.winner = 0
                    self.gameOver = True
                else:
                    self.winner = 1
                    self.gameOver = True
            
        if " " not in self.board:
            self.gameOver = True
            self.winner = 2
        return
    
    def reset(self):
        self.board = [" " for x in range(9)]
        self.gameOver = False
        self.winner = ""

class player:
     def __init__(self, name, strategy, epsilon):
        self.name = name
        self.strategy = strategy
        self.epsilon = epsilon
        self.q_table = {}
        self.states = []
        
     def makeMove(self, board):
        validMoves = [] 
        for x in range(len(board)):
            if(board[x] == " "):
                validMoves.append(x)
        if(self.strategy == "human"):
            print("Your valid moves are:", validMoves)
            print("Where would you like to move?")
            move = int(input())
            return move
        elif(self.strategy == "random"):
            return random.choice(validMoves)
        elif(self.strategy == "AI"):
            state = tuple(board)
            move_index = 0
            if(state not in self.q_table):
                self.q_table[state] = [0 for x in range(len(validMoves))]
            if(random.random() < self.epsilon):
                move_index = random.randint(0, len(validMoves)-1)
                move = validMoves[move_index]
            else:
                move_index = np.argmax(self.q_table[state])
                move = validMoves[move_index]
            self.states.append((state, move_index))
            return move
        
        def updateQTable(self, reward):
             alpha = 0.1  
             gamma = 0.9 
             for (state, action_index) in self.states:
                max_future_q = max(self.q_table[state])  
                self.q_table[state][action_index] += alpha * (reward + gamma * max_future_q - self.q_table[state][action_index])
                self.states.clear()
   


               


def main():
    print("Tic-tac-toe game started!")

if __name__ == "__main__":
    main()

