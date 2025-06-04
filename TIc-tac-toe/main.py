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


               


def main():
    print("Tic-tac-toe game started!")

if __name__ == "__main__":
    main()

