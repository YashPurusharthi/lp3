import heapq

class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    def __lt__(self, nxt):
        return self.freq < nxt.freq

def printNodes(node, val=''):
    newVal = val + str(node.huff)
    if node.left:
        printNodes(node.left, newVal)
    if node.right:
        printNodes(node.right, newVal)
    if not node.left and not node.right:
        print(f"{node.symbol} -> {newVal}")

# Prompt the user to enter a string
#string = input("Enter a string: ")

# Create a dictionary to store character frequencies
#count = {}

# Count character frequencies in the input string
#for char in string:
#    if char not in count:
#        count[char] = 0
#    count[char] += 1

# Convert character frequencies to the required format
#chars = list(count.keys())
#freq = list(count.values())

# characters for huffman tree 
chars = ['a', 'b', 'c', 'd', 'e'] 

# frequency of characters 
freq = [7, 2, 6, 3, 9] 

# List containing unused nodes
nodes = []

# Convert characters and frequencies into Huffman tree nodes
for x in range(len(chars)):
    heapq.heappush(nodes, Node(freq[x], chars[x]))

while len(nodes) > 1:
    left = heapq.heappop(nodes)
    right = heapq.heappop(nodes)
    left.huff = 0
    right.huff = 1
    newNode = Node(left.freq + right.freq, left.symbol + right.symbol, left, right)
    heapq.heappush(nodes, newNode)

# Huffman Tree is ready!
print("Huffman Codes:")
printNodes(nodes[0])
