def solveNQueens(n):
    board = [ ['.' for x in range(n)] for x in range(n)]
    solutions = []

    def isSafe(row, col):
        
        # Check if there is a queen in the same column
        for i in range(row):
            if board[i][col] == 'Q':
                return False

            
        # Check the upper-left diagonal
        i = row - 1
        j = col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i = i - 1
            j = j - 1

            
        # Check the upper-right diagonal
        i = row - 1
        j = col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i = i - 1
            j = j + 1

        return True

    
    
    def backtrack(row):
        if row == n:
            # Found a valid solution, add it to the list of solutions
            solutions.append([' '.join(row) for row in board])
            return

        for col in range(n):
            if isSafe(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'

    backtrack(0)
    return solutions


solutions = solveNQueens(4)
print("possible solution",len(solutions))
for solution in solutions:
    for row in solution:
        print(row)
    print()
