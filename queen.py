def is_safe(board, row, col):
    # Check the row on the left side
    for i in range(col):
        if board[row][i] == 1:
            return False

    # Check upper diagonal on the left side
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check lower diagonal on the left side
    for i, j in zip(range(row, len(board), 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens(board, col):
    if col == len(board):
        return True

    for i in range(len(board)):
        if is_safe(board, i, col):
            board[i][col] = 1
            if solve_n_queens(board, col + 1):
                return True
            board[i][col] = 0  # Backtrack if placement leads to no solution

    return False

def print_board(board):
    for row in board:
        print(" ".join(["Q" if cell == 1 else "-" for cell in row]))

# Create an 8x8 chessboard with all cells initialized to 0
n = 8
chessboard = [[0 for _ in range(n)] for _ in range(n)]

# Place the first queen in the first row, first column
chessboard[0][0] = 1

# Try to solve the remaining queens
if solve_n_queens(chessboard, 1):
    print("Solution found:")
    print_board(chessboard)
else:
    print("No solution exists.")