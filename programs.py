# Tic Tac Toe prblm

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def check_winner(board, player):
    for row in board:
        if all(cell == player for cell in row):
            return True
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def is_board_full(board):
    return all(cell != " " for row in board for cell in row)

def main():
    board = [[" " for _ in range(3)] for _ in range(3)]
    current_player = "X"

    print("Welcome to Tic Tac Toe!")
    print_board(board)

    while True:
        row = int(input(f"Player {current_player}, enter row (0, 1, 2): "))
        col = int(input(f"Player {current_player}, enter column (0, 1, 2): "))

        if board[row][col] == " ":
            board[row][col] = current_player
            print_board(board)

            if check_winner(board, current_player):
                print(f"Player {current_player} wins!")
                break
            elif is_board_full(board):
                print("It's a draw!")
                break

            current_player = "O" if current_player == "X" else "X"
        else:
            print("That cell is already taken. Try again.")

# Call the main function to start the game
main()



#Python program to illustrate Missionaries & cannibals Problem
#This code is contributed by Sunit Mal
print("\n")
print("\tGame Start\nNow the task is to move all of them to right side of the river")
print("rules:\n1. The boat can carry at most two people\n2. If cannibals num greater than missionaries then the cannibals would eat the missionaries\n3. The boat cannot cross the river by itself with no people on board")
lM = 3		 #lM = Left side Missionaries number
lC = 3		 #lC = Laft side Cannibals number
rM=0		 #rM = Right side Missionaries number
rC=0		 #rC = Right side cannibals number
userM = 0	 #userM = User input for number of missionaries for right to left side travel
userC = 0	 #userC = User input for number of cannibals for right to left travel
k = 0
print("\nM M M C C C |	 --- | \n")
try:
	while(True):
		while(True):
			print("Left side -> right side river travel")
			#uM = user input for number of missionaries for left to right travel
			#uC = user input for number of cannibals for left to right travel
			uM = int(input("Enter number of Missionaries travel => "))	
			uC = int(input("Enter number of Cannibals travel => "))

			if((uM==0)and(uC==0)):
				print("Empty travel not possible")
				print("Re-enter : ")
			elif(((uM+uC) <= 2)and((lM-uM)>=0)and((lC-uC)>=0)):
				break
			else:
				print("Wrong input re-enter : ")
		lM = (lM-uM)
		lC = (lC-uC)
		rM += uM
		rC += uC

		print("\n")
		for i in range(0,lM):
			print("M ",end="")
		for i in range(0,lC):
			print("C ",end="")
		print("| --> | ",end="")
		for i in range(0,rM):
			print("M ",end="")
		for i in range(0,rC):
			print("C ",end="")
		print("\n")

		k +=1

		if(((lC==3)and (lM == 1))or((lC==3)and(lM==2))or((lC==2)and(lM==1))or((rC==3)and (rM == 1))or((rC==3)and(rM==2))or((rC==2)and(rM==1))):
			print("Cannibals eat missionaries:\nYou lost the game")

			break

		if((rM+rC) == 6):
			print("You won the game : \n\tCongrats")
			print("Total attempt")
			print(k)
			break
		while(True):
			print("Right side -> Left side river travel")
			userM = int(input("Enter number of Missionaries travel => "))
			userC = int(input("Enter number of Cannibals travel => "))
			
			if((userM==0)and(userC==0)):
					print("Empty travel not possible")
					print("Re-enter : ")
			elif(((userM+userC) <= 2)and((rM-userM)>=0)and((rC-userC)>=0)):
				break
			else:
				print("Wrong input re-enter : ")
		lM += userM
		lC += userC
		rM -= userM
		rC -= userC

		k +=1
		print("\n")
		for i in range(0,lM):
			print("M ",end="")
		for i in range(0,lC):
			print("C ",end="")
		print("| <-- | ",end="")
		for i in range(0,rM):
			print("M ",end="")
		for i in range(0,rC):
			print("C ",end="")
		print("\n")

	

		if(((lC==3)and (lM == 1))or((lC==3)and(lM==2))or((lC==2)and(lM==1))or((rC==3)and (rM == 1))or((rC==3)and(rM==2))or((rC==2)and(rM==1))):
			print("Cannibals eat missionaries:\nYou lost the game")
			break
except EOFError as e:
	print("\nInvalid input please retry !!")


from collections import deque

class State:
    def __init__(self, lion, goat, grass, farmer, parent=None):
        self.lion = lion
        self.goat = goat
        self.grass = grass
        self.farmer = farmer
        self.parent = parent

    def __eq__(self, other):
        return (self.lion, self.goat, self.grass, self.farmer) == (other.lion, other.goat, other.grass, other.farmer)

    def __hash__(self):
        return hash((self.lion, self.goat, self.grass, self.farmer))

    def is_valid(self):
        if (self.goat == self.lion and self.farmer != self.goat) or (self.goat == self.grass and self.farmer != self.goat):
            return False
        return True

    def is_final(self):
        return self.lion == 'right' and self.goat == 'right' and self.grass == 'right' and self.farmer == 'right'

    def __repr__(self):
        return f"Lion: {self.lion}, Goat: {self.goat}, Grass: {self.grass}, Farmer: {self.farmer}"

def generate_moves():
    return ['Lion', 'Goat', 'Grass', 'Farmer']

def apply_move(state, move):
    new_state = State(state.lion, state.goat, state.grass, state.farmer)
    if move == 'Lion':
        new_state.lion = 'right' if state.lion == 'left' else 'left'
        new_state.farmer = 'right' if state.farmer == 'left' else 'left'
    elif move == 'Goat':
        new_state.goat = 'right' if state.goat == 'left' else 'left'
        new_state.farmer = 'right' if state.farmer == 'left' else 'left'
    elif move == 'Grass':
        new_state.grass = 'right' if state.grass == 'left' else 'left'
        new_state.farmer = 'right' if state.farmer == 'left' else 'left'
    elif move == 'Farmer':
        new_state.farmer = 'right' if state.farmer == 'left' else 'left'
    return new_state

def solve_game():
    initial_state = State('left', 'left', 'left', 'left')
    visited_states = set()
    queue = deque([initial_state])

    while queue:
        current_state = queue.popleft()
        if current_state.is_final():
            solution = []
            while current_state:
                solution.insert(0, current_state)
                current_state = current_state.parent
            return solution
        visited_states.add(current_state)

        for move in generate_moves():
            new_state = apply_move(current_state, move)
            if new_state.is_valid() and new_state not in visited_states:
                new_state.parent = current_state
                queue.append(new_state)

    return None

solution = solve_game()

if solution:
    for index, state in enumerate(solution):
        print("Step {}: Lion: {}, Goat: {}, Grass: {}, Farmer: {}".format(
            index + 1, state.lion, state.goat, state.grass, state.farmer))
else:
    print("No solution found.")



# jealous husband problem 


people_dict = {"h1": 0, "h2": 0, "h3": 0, "w1": 0, "w2": 0, "w3": 0}

people_names = list(people_dict.keys())

boat = 0

 

def alterbit(bit):

    return abs(bit - 1)

 

def jealousy(d):

    for i in [1, 2, 3]:

        if d["h" + str(i)] != d["w" + str(i)]:

            if (d["h1"] == d["w" + str(i)]) or (d["h2"] == d["w" + str(i)]) or (d["h3"] == d["w" + str(i)]):

                return 1

    return 0

 

def turn():

    global boat

    good_people = []

    for i in people_names:

        if people_dict[i] == boat:

            good_people.append(i)

    print("Available to choose from: " + ', '.join(good_people))

    move = input("Choose your move. ")

    boat_other = 0

    for i in people_names:

        if i in move:

            if not (i in good_people):

                boat_other = 1

                print("Cannot move " + i + "; the boat is on the wrong side!\n")

                return 0

    counter = 0

    for i in people_names:

        if i in move:

            counter += 1

    if counter > 2:

        print("Cannot move more than two!\n")

        return 0

    temp_people_dict = dict(people_dict)

    for i in people_names:

        if i in move:

            temp_people_dict[i] = alterbit(temp_people_dict[i])

    if jealousy(temp_people_dict):

        print("A wife cannot be left with another man unless her husband is present.")

        return 0

    counter = 0

    for i in people_names:

        if i in move:

            people_dict[i] = alterbit(people_dict[i])

            counter += 1

    if counter != 0:

        boat = alterbit(boat)

 

turn_number = 0

while not (sum(people_dict[i] for i in people_names) == 6):

    turn()

    turn_number += 1

print("Congratulations! You won in " + str(turn_number) + " turns. (Minimum 11)")



from collections import deque

def get_next_states(state):
    next_states = []
    empty_row, empty_col = None, None

    
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                empty_row, empty_col = i, j
                break

    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for move in moves:
        new_row = empty_row + move[0]
        new_col = empty_col + move[1]
        
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            new_state = [list(row) for row in state]
            new_state[empty_row][empty_col], new_state[new_row][new_col] = new_state[new_row][new_col], new_state[empty_row][empty_col]
            next_states.append(tuple(tuple(row) for row in new_state))
    
    return next_states

def solve_puzzle(initial_state, goal_state):
    visited = set()
    queue = deque([(initial_state, [])])
    
    while queue:
        current_state, path = queue.popleft()
        visited.add(current_state)
        
        if current_state == goal_state:
            return path
        
        for next_state in get_next_states(current_state):
            if next_state not in visited:
                queue.append((next_state, path + [next_state]))

    return None


print("Enter the initial state of the puzzle:")
initial_state = []
for _ in range(3):
    row = input().split()
    initial_state.append([int(tile) for tile in row])

goal_state = ((1, 2, 3), (8, 0, 4), (7, 6, 5))


solution = solve_puzzle(tuple(tuple(row) for row in initial_state), goal_state)

if solution:
    print("Solution found!")
    for step, state in enumerate(solution):
        print(f"Step {step}:\n")
        for row in state:
            print(row)
else:
    print("No solution found.")



# Travelling sales person prblem using DFS

import math

def tsp_dp(graph):
    n = len(graph)
    memo = {}

    def dfs(mask, current):
        if mask == (1 << n) - 1:
            return graph[current][0]
        
        if (mask, current) in memo:
            return memo[(mask, current)]
        
        min_distance = math.inf
        for next_node in range(n):
            if mask & (1 << next_node) == 0:
                new_mask = mask | (1 << next_node)
                distance = graph[current][next_node] + dfs(new_mask, next_node)
                min_distance = min(min_distance, distance)
        
        memo[(mask, current)] = min_distance
        return min_distance
    
    return dfs(1, 0)

def main():
    n = int(input("Enter the number of cities: "))
    graph = []

    print("Enter the distance matrix:")
    for _ in range(n):
        row = list(map(int, input().split()))
        graph.append(row)

    min_distance = tsp_dp(graph)
    print("Minimum Distance:", min_distance)

if _name_ == "_main_":
    main()


# Travelling sales person prblm using BFS

from collections import deque
import math

def tsp_bfs(graph, start):
    n = len(graph)
    visited = [[False] * n for _ in range(1 << n)]
    queue = deque([(start, 1 << start, 0)])  # (current_city, visited_mask, total_distance)
    min_distance = math.inf

    while queue:
        current, mask, distance = queue.popleft()

        if mask == (1 << n) - 1:
            min_distance = min(min_distance, distance + graph[current][start])
        
        for next_city in range(n):
            if not visited[mask][next_city]:
                new_mask = mask | (1 << next_city)
                new_distance = distance + graph[current][next_city]
                queue.append((next_city, new_mask, new_distance))
                visited[new_mask][next_city] = True

    return min_distance

def main():
    n = int(input("Enter the number of cities: "))
    graph = []

    print("Enter the distance matrix:")
    for _ in range(n):
        row = list(map(int, input().split()))
        graph.append(row)
    
    start = int(input("Enter the starting city (0 to {}): ".format(n - 1)))
    
    if start < 0 or start >= n:
        print("Invalid starting city. Please enter a valid starting city.")
        return

    min_distance = tsp_bfs(graph, start)
    print("Minimum Distance:", min_distance)

if _name_ == "_main_":
    main()



# 8 puzzle problm using A* algorithm

import heapq
import copy

class Node:
    def _init_(self, state, parent=None, g=0, h=0):
        self.state = state
        self.parent = parent
        self.g = g
        self.h = h

    def _lt_(self, other):
        return (self.g + self.h) < (other.g + other.h)

def print_puzzle(puzzle):
    for row in puzzle:
        print(" ".join(map(str, row)))

def find_blank(puzzle):
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] == 0:
                return i, j

def is_goal(state, goal_state):
    return state == goal_state

def misplaced_tiles(state, goal_state):
    count = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != goal_state[i][j]:
                count += 1
    return count

def get_neighbors(state):
    i, j = find_blank(state)
    neighbors = []

    if i > 0:
        up_neighbor = copy.deepcopy(state)
        up_neighbor[i][j], up_neighbor[i - 1][j] = up_neighbor[i - 1][j], up_neighbor[i][j]
        neighbors.append(up_neighbor)

    if i < 2:
        down_neighbor = copy.deepcopy(state)
        down_neighbor[i][j], down_neighbor[i + 1][j] = down_neighbor[i + 1][j], down_neighbor[i][j]
        neighbors.append(down_neighbor)

    if j > 0:
        left_neighbor = copy.deepcopy(state)
        left_neighbor[i][j], left_neighbor[i][j - 1] = left_neighbor[i][j - 1], left_neighbor[i][j]
        neighbors.append(left_neighbor)

    if j < 2:
        right_neighbor = copy.deepcopy(state)
        right_neighbor[i][j], right_neighbor[i][j + 1] = right_neighbor[i][j + 1], right_neighbor[i][j]
        neighbors.append(right_neighbor)

    return neighbors

def astar_search(initial_state, goal_state):
    open_set = []
    closed_set = set()

    initial_node = Node(initial_state)
    heapq.heappush(open_set, initial_node)

    while open_set:
        current_node = heapq.heappop(open_set)

        if is_goal(current_node.state, goal_state):
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(tuple(map(tuple, current_node.state)))

        for neighbor_state in get_neighbors(current_node.state):
            if tuple(map(tuple, neighbor_state)) in closed_set:
                continue

            g_score = current_node.g + 1
            h_score = misplaced_tiles(neighbor_state, goal_state)
            neighbor_node = Node(neighbor_state, current_node, g_score, h_score)

            if neighbor_node not in open_set:
                heapq.heappush(open_set, neighbor_node)

    return None

def input_puzzle(prompt):
    print(prompt)
    puzzle = []
    for i in range(3):
        row = list(map(int, input().split()))
        puzzle.append(row)
    return puzzle

if _name_ == "_main_":
    print("Enter the initial state (3x3 puzzle, use 0 for the blank tile):")
    initial_state = input_puzzle("")

    print("Enter the goal state (3x3 puzzle, use 0 for the blank tile):")
    goal_state = input_puzzle("")

    path = astar_search(initial_state, goal_state)

    if path:
        print("Solution found:")
        for step, state in enumerate(path):
            print(f"Step {step + 1}:")
            print_puzzle(state)
    else:
        print("No solution found.")




#water jug prblm using BFS algorithm

from collections import deque

class State:
    def _init_(self, x, y):
        self.x = x  
        self.y = y  

    def _eq_(self, other):
        return self.x == other.x and self.y == other.y

    def _hash_(self):
        return hash((self.x, self.y))

    def _str_(self):
        return f"(jug 1:{self.x}, jug 2:{self.y})"

def water_jug_bfs(jug_x_capacity, jug_y_capacity, target_amount):
    visited = set()
    queue = deque([(State(0, 0), [])])  

    while queue:
        current_state, path = queue.popleft()

        if current_state in visited:
            continue

        visited.add(current_state)

        if current_state.x == target_amount or current_state.y == target_amount:
            path.append(current_state)
            return path

        # Fill jug X
        if current_state.x < jug_x_capacity:
            new_x = jug_x_capacity
            new_y = current_state.y
            if State(new_x, new_y) not in visited:
                queue.append((State(new_x, new_y), path + [current_state]))

        # Fill jug Y
        if current_state.y < jug_y_capacity:
            new_x = current_state.x
            new_y = jug_y_capacity
            if State(new_x, new_y) not in visited:
                queue.append((State(new_x, new_y), path + [current_state]))

        # Empty jug X
        if current_state.x > 0:
            new_x = 0
            new_y = current_state.y
            if State(new_x, new_y) not in visited:
                queue.append((State(new_x, new_y), path + [current_state]))

        # Empty jug Y
        if current_state.y > 0:
            new_x = current_state.x
            new_y = 0
            if State(new_x, new_y) not in visited:
                queue.append((State(new_x, new_y), path + [current_state]))

        # Pour from X to Y
        if current_state.x > 0 and current_state.y < jug_y_capacity:
            pour_amount = min(current_state.x, jug_y_capacity - current_state.y)
            new_x = current_state.x - pour_amount
            new_y = current_state.y + pour_amount
            if State(new_x, new_y) not in visited:
                queue.append((State(new_x, new_y), path + [current_state]))

        # Pour from Y to X
        if current_state.y > 0 and current_state.x < jug_x_capacity:
            pour_amount = min(current_state.y, jug_x_capacity - current_state.x)
            new_x = current_state.x + pour_amount
            new_y = current_state.y - pour_amount
            if State(new_x, new_y) not in visited:
                queue.append((State(new_x, new_y), path + [current_state]))

    return None

def get_user_input():
    jug_x_capacity = int(input("Enter the capacity of jug X: "))
    jug_y_capacity = int(input("Enter the capacity of jug Y: "))
    target_amount = int(input("Enter the target amount of water: "))
    return jug_x_capacity, jug_y_capacity, target_amount

def main():
    jug_x_capacity, jug_y_capacity, target_amount = get_user_input()

    solution = water_jug_bfs(jug_x_capacity, jug_y_capacity, target_amount)

    if solution:
        print("Solution Found:")
        for step, state in enumerate(solution):
            print(f"Step {step + 1}: {state}")
    else:
        print("No solution found.")

if _name_ == "_main_":
    main()




#water jug prblm using dfs method

class State:
    def _init_(self, jug1, jug2):
        self.jug1 = jug1  # Current water amount in jug1
        self.jug2 = jug2  # Current water amount in jug2

    def _eq_(self, other):
        return self.jug1 == other.jug1 and self.jug2 == other.jug2

    def _hash_(self):
        return hash((self.jug1, self.jug2))

    def _str_(self):
        return f"(Jug1: {self.jug1}, Jug2: {self.jug2})"


def water_jug_dfs(jug1_capacity, jug2_capacity, jug1_target, visited, path):
    current_state = path[-1]

    if current_state in visited:
        return None

    visited.add(current_state)

    if current_state.jug1 == jug1_target:
        return path

    possible_moves = []

    # Fill Jug1
    if current_state.jug1 < jug1_capacity:
        possible_moves.append(State(jug1_capacity, current_state.jug2))

    # Fill Jug2
    if current_state.jug2 < jug2_capacity:
        possible_moves.append(State(current_state.jug1, jug2_capacity))

    # Empty Jug1
    if current_state.jug1 > 0:
        possible_moves.append(State(0, current_state.jug2))

    # Empty Jug2
    if current_state.jug2 > 0:
        possible_moves.append(State(current_state.jug1, 0))

    # Pour water from Jug1 to Jug2
    if current_state.jug1 > 0 and current_state.jug2 < jug2_capacity:
        pour_amount = min(current_state.jug1, jug2_capacity - current_state.jug2)
        possible_moves.append(State(current_state.jug1 - pour_amount, current_state.jug2 + pour_amount))

    # Pour water from Jug2 to Jug1
    if current_state.jug2 > 0 and current_state.jug1 < jug1_capacity:
        pour_amount = min(current_state.jug2, jug1_capacity - current_state.jug1)
        possible_moves.append(State(current_state.jug1 + pour_amount, current_state.jug2 - pour_amount))

    for move in possible_moves:
        new_path = path + [move]
        result = water_jug_dfs(jug1_capacity, jug2_capacity, jug1_target, visited, new_path)
        if result:
            return result

    return None


def get_user_input():
    jug1_capacity = int(input("Enter the capacity of Jug1: "))
    jug2_capacity = int(input("Enter the capacity of Jug2: "))
    jug1_target = int(input("Enter the target amount for Jug1: "))
    return jug1_capacity, jug2_capacity, jug1_target


def main():
    jug1_capacity, jug2_capacity, jug1_target = get_user_input()

    initial_state = State(0, 0)
    visited = set()
    path = [initial_state]

    solution = water_jug_dfs(jug1_capacity, jug2_capacity, jug1_target, visited, path)

    if solution:
        print("Solution Found:")
        for step, state in enumerate(solution):
            print(f"Step {step + 1}: {state}")
    else:
        print("No solution found.")


if _name_ == "_main_":
    main()




N = int(input("Enter the number of queens (N):"))
ld = [0] * (2 * N - 1)
rd = [0] * (2 * N - 1)
cl = [0] * N
def printSolution(board):
    for i in range(N):
        for j in range(N):
            print(" Q " if board[i][j] == 1 else " . ", end="")
        print()
def solveNQUtil(board, col):
    if col >= N:
        return True
    for i in range(N):
        if (ld[i - col + N - 1] != 1 and rd[i + col] != 1) and cl[i] != 1:
            board[i][col] = 1
            ld[i - col + N - 1] = rd[i + col] = cl[i] = 1
            if solveNQUtil(board, col + 1):
                return True
            board[i][col] = 0 
            ld[i - col + N - 1] = rd[i + col] = cl[i] = 0
    return False
def solveNQ():
    board = [[0 for _ in range(N)] for _ in range(N)]

    if not solveNQUtil(board, 0):
        print("Solution does not exist")
        return False

    print("A solution to the N-Queens problem for N =", N, "is:")
    printSolution(board)
    return True
if __name__ == "__main__":
    solveNQ()




def celsius_to_fahrenheit(celsius):
    fahrenheit = (celsius * 9/5) + 32
    return fahrenheit

# Get user input for Celsius temperature
celsius_temp = float(input("Enter a temperature in degrees Celsius: "))

# Get user input for the threshold temperature
threshold_temp = float(input("Enter a threshold temperature in degrees Fahrenheit: "))

# Convert Celsius temperature to Fahrenheit
fahrenheit_temp = celsius_to_fahrenheit(celsius_temp)

# Output the conversion result
print(f"{celsius_temp}°C is equivalent to {fahrenheit_temp}°F")

# Check if the temperature is below the threshold
if fahrenheit_temp < threshold_temp:
    print(f"{fahrenheit_temp}°F is below {threshold_temp}°F")
else:
    print(f"{fahrenheit_temp}°F is not below {threshold_temp}°F")



class State:
    def __init__(self, monkey_position, box_position, banana_position, on_box):
        self.monkey_position = monkey_position
        self.box_position = box_position
        self.banana_position = banana_position
        self.on_box = on_box

    def __eq__(self, other):
        return (self.monkey_position == other.monkey_position and
                self.box_position == other.box_position and
                self.banana_position == other.banana_position and
                self.on_box == other.on_box)

    def __hash__(self):
        return hash((self.monkey_position, self.box_position, self.banana_position, self.on_box))

def monkey_banana_problem(initial_state, goal_state):
    visited = set()
    stack = [([], initial_state)]

    while stack:
        actions, current_state = stack.pop()

        if current_state == goal_state:
            return actions

        visited.add(current_state)

        if current_state.monkey_position == 'left':
            new_state = State('right', current_state.box_position, current_state.banana_position, current_state.on_box)
            if new_state not in visited:
                stack.append((actions + ['Move right'], new_state))

            if current_state.on_box:
                new_state = State('right', current_state.box_position, current_state.banana_position, False)
                if new_state not in visited:
                    stack.append((actions + ['Climb onto the box'], new_state))

        elif current_state.monkey_position == 'right':
            new_state = State('left', current_state.box_position, current_state.banana_position, current_state.on_box)
            if new_state not in visited:
                stack.append((actions + ['Move left'], new_state))

            if current_state.on_box:
                new_state = State('left', current_state.box_position, current_state.banana_position, False)
                if new_state not in visited:
                    stack.append((actions + ['Climb onto the box'], new_state))

        if current_state.monkey_position == current_state.box_position:
            if current_state.on_box:
                new_state = State(current_state.monkey_position, 'right', current_state.banana_position, True)
                if new_state not in visited:
                    stack.append((actions + ['Push the box right'], new_state))
            else:
                new_state = State(current_state.monkey_position, 'left', current_state.banana_position, True)
                if new_state not in visited:
                    stack.append((actions + ['Push the box left'], new_state))

    return None

def main():
    try:
        monkey_position = input("Enter the initial position of the monkey (left or right): ")
        box_position = input("Enter the initial position of the box (left or right): ")
        banana_position = input("Enter the initial position of the banana (left or right): ")

        initial_state = State(monkey_position, box_position, banana_position, False)
        goal_state = State('right', 'right', 'right', True)

        solution = monkey_banana_problem(initial_state, goal_state)
        if solution:
            print("Solution:")
            for action in solution:
                print(action)
        else:
            print("No solution found.")
    except KeyboardInterrupt:
        print("\nOperation interrupted.")

if __name__ == "__main__":
    main()
