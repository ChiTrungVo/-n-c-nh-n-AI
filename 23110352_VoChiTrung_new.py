import pygame
import sys
import copy
import time
import random
import math
from collections import deque
import heapq

pygame.init()

WIDTH, HEIGHT = 800, 750
GRID_SIZE = 100
FONT = pygame.font.SysFont(None, 48)

WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
BLUE = (100, 149, 237)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8 Puzzle Solver")

# BFS Solver
def bfs(start):
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    visited = set()
    queue = deque()
    queue.append((start, []))

    def serialize(state):
        return tuple(tuple(row) for row in state)

    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves

    start_time = time.time()
    while queue:
        current, path = queue.popleft()
        ser = serialize(current)
        if ser in visited:
            continue
        visited.add(ser)
        if current == goal:
            elapsed_time = time.time() - start_time
            return path + [current], len(path), elapsed_time, True
        for neighbor in get_neighbors(current):
            queue.append((neighbor, path + [current]))
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

# DFS Solver
def dfs(start):
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    visited = set()
    stack = [(start, [])]

    def serialize(state):
        return tuple(tuple(row) for row in state)

    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves

    start_time = time.time()
    while stack:
        current, path = stack.pop()
        ser = serialize(current)
        if ser in visited:
            continue
        visited.add(ser)
        if current == goal:
            elapsed_time = time.time() - start_time
            return path + [current], len(path), elapsed_time, True
        for neighbor in get_neighbors(current):
            stack.append((neighbor, path + [current]))
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

# Uniform Cost Search (UCS) Solver
def ucs(start):
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    visited = set()
    queue = deque([(start, [], 0)])  # (state, path, cost)
    
    def serialize(state):
        return tuple(tuple(row) for row in state)

    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves
    
    start_time = time.time()
    while queue:
        current, path, cost = queue.popleft()
        ser = serialize(current)
        if ser in visited:
            continue
        visited.add(ser)
        if current == goal:
            elapsed_time = time.time() - start_time
            return path + [current], len(path), elapsed_time, True
        for neighbor in get_neighbors(current):
            queue.append((neighbor, path + [current], cost + 1))  # Cost for each step is 1
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

# Iterative Deepening DFS (IDS) Solver
def ids(start):
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    
    def serialize(state):
        return tuple(tuple(row) for row in state)

    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves

    def dfs_limited(state, depth, path):
        if depth == 0:
            return None
        if state == goal:
            return path + [state]
        for neighbor in get_neighbors(state):
            result = dfs_limited(neighbor, depth - 1, path + [state])
            if result is not None:
                return result
        return None

    start_time = time.time()
    depth = 1
    while True:
        result = dfs_limited(start, depth, [])
        if result is not None:
            elapsed_time = time.time() - start_time
            return result, len(result), elapsed_time, True
        depth += 1
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

def manhattan_heuristic(state, goal):
    """Calculate Manhattan distance between current and goal state."""
    distance = 0
    for i in range(3):
        for j in range(3):
            val = state[i][j]
            if val != 0:
                goal_x, goal_y = divmod(val - 1, 3)
                distance += abs(goal_x - i) + abs(goal_y - j)
    return distance

def greedy(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves
    
    start_time = time.time()
    visited = set()
    queue = []
    heapq.heappush(queue, (manhattan_heuristic(start, goal), start, []))
    
    while queue:
        _, current, path = heapq.heappop(queue)
        ser = serialize(current)
        if ser in visited:
            continue
        visited.add(ser)
        if current == goal:
            elapsed_time = time.time() - start_time
            return path + [current], len(path), elapsed_time, True
        for neighbor in get_neighbors(current):
            heapq.heappush(queue, (manhattan_heuristic(neighbor, goal), neighbor, path + [current]))
    
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

def a_star(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves
    
    start_time = time.time()
    visited = set()
    queue = []
    heapq.heappush(queue, (manhattan_heuristic(start, goal), 0, start, []))  # (f, g, state, path)
    
    while queue:
        _, g, current, path = heapq.heappop(queue)
        ser = serialize(current)
        if ser in visited:
            continue
        visited.add(ser)
        if current == goal:
            elapsed_time = time.time() - start_time
            return path + [current], len(path), elapsed_time, True
        for neighbor in get_neighbors(current):
            f = g + 1 + manhattan_heuristic(neighbor, goal)  # f = g + h
            heapq.heappush(queue, (f, g + 1, neighbor, path + [current]))
    
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

def ida_star(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves
    
    def dfs_limited(state, g, limit, path):
        f = g + manhattan_heuristic(state, goal)
        if f > limit:
            return f
        if state == goal:
            return path + [state]
        min_cost = float('inf')
        for neighbor in get_neighbors(state):
            result = dfs_limited(neighbor, g + 1, limit, path + [state])
            if isinstance(result, list):
                return result
            min_cost = min(min_cost, result)
        return min_cost
    
    start_time = time.time()
    limit = manhattan_heuristic(start, goal)
    while True:
        result = dfs_limited(start, 0, limit, [])
        if isinstance(result, list):
            elapsed_time = time.time() - start_time
            return result, len(result), elapsed_time, True
        limit = result
    
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

# Simple Hill Climbing Solver
def hill_climbing(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves
    
    start_time = time.time()
    current = start
    path = [current]
    visited = set()
    visited.add(serialize(current))
    
    while current != goal:
        neighbors = get_neighbors(current)
        best_neighbor = None
        best_heuristic = float('inf')
        
        for neighbor in neighbors:
            if serialize(neighbor) not in visited:
                h = manhattan_heuristic(neighbor, goal)
                if h < best_heuristic:
                    best_heuristic = h
                    best_neighbor = neighbor
        
        if best_neighbor is None:
            elapsed_time = time.time() - start_time
            return path, len(path) - 1, elapsed_time, False
        
        current = best_neighbor
        path.append(current)
        visited.add(serialize(current))
    
    elapsed_time = time.time() - start_time
    return path, len(path) - 1, elapsed_time, True

# Steepest-Ascent Hill Climbing Solver
def steepest_ascent_hill_climbing(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves
    
    start_time = time.time()
    current = start
    path = [current]
    visited = set()
    visited.add(serialize(current))
    
    while current != goal:
        neighbors = get_neighbors(current)
        best_neighbor = None
        best_heuristic = manhattan_heuristic(current, goal)
        
        for neighbor in neighbors:
            if serialize(neighbor) not in visited:
                h = manhattan_heuristic(neighbor, goal)
                if h < best_heuristic:
                    best_heuristic = h
                    best_neighbor = neighbor
        
        if best_neighbor is None:
            elapsed_time = time.time() - start_time
            return path, len(path) - 1, elapsed_time, False
        
        current = best_neighbor
        path.append(current)
        visited.add(serialize(current))
    
    elapsed_time = time.time() - start_time
    return path, len(path) - 1, elapsed_time, True

# Stochastic Hill Climbing Solver
def stochastic_hill_climbing(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves
    
    start_time = time.time()
    current = start
    path = [current]
    visited = set()
    visited.add(serialize(current))
    
    while current != goal:
        neighbors = get_neighbors(current)
        valid_neighbors = [(n, manhattan_heuristic(n, goal)) for n in neighbors if serialize(n) not in visited]
        
        if not valid_neighbors:
            elapsed_time = time.time() - start_time
            return path, len(path) - 1, elapsed_time, False
        
        total_weight = sum(max(0, manhattan_heuristic(current, goal) - h) for _, h in valid_neighbors)
        if total_weight == 0:
            neighbor = random.choice([n for n, _ in valid_neighbors])
        else:
            weights = [max(0, manhattan_heuristic(current, goal) - h) for _, h in valid_neighbors]
            neighbor = random.choices([n for n, _ in valid_neighbors], weights=weights, k=1)[0]
        
        current = neighbor
        path.append(current)
        visited.add(serialize(current))
    
    elapsed_time = time.time() - start_time
    return path, len(path) - 1, elapsed_time, True

# Simulated Annealing Solver
def simulated_annealing(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves
    
    start_time = time.time()
    current = start
    path = [current]
    temperature = 1000
    cooling_rate = 0.995
    min_temperature = 0.01
    
    while current != goal and temperature > min_temperature:
        neighbors = get_neighbors(current)
        if not neighbors:
            elapsed_time = time.time() - start_time
            return path, len(path) - 1, elapsed_time, False
        
        next_state = random.choice(neighbors)
        current_h = manhattan_heuristic(current, goal)
        next_h = manhattan_heuristic(next_state, goal)
        delta_h = next_h - current_h
        
        if delta_h <= 0 or random.random() < math.exp(-delta_h / temperature):
            current = next_state
            path.append(current)
        
        temperature *= cooling_rate
    
    elapsed_time = time.time() - start_time
    return path, len(path) - 1, elapsed_time, current == goal

# Genetic Algorithm Solver
def genetic_algorithm(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves
    
    def mutate(state):
        neighbors = get_neighbors(state)
        return random.choice(neighbors) if neighbors else state
    
    def crossover(parent1, parent2):
        child = copy.deepcopy(parent1)
        for i in range(3):
            if random.random() < 0.5:
                child[i] = copy.deepcopy(parent2[i])
        flat = [num for row in child for num in row]
        if len(set(flat)) != 9:
            return parent1
        return child
    
    start_time = time.time()
    population_size = 50
    generations = 1000
    mutation_rate = 0.1
    
    population = [copy.deepcopy(start)]
    for _ in range(population_size - 1):
        state = copy.deepcopy(start)
        for _ in range(random.randint(1, 10)):
            state = mutate(state)
        population.append(state)
    
    for _ in range(generations):
        population = sorted(population, key=lambda x: manhattan_heuristic(x, goal))
        
        if population[0] == goal:
            path = [population[0]]
            elapsed_time = time.time() - start_time
            return path, 0, elapsed_time, True
        
        new_population = population[:10]
        
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population[:20], 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child)
            new_population.append(child)
        
        population = new_population
    
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

# Beam Search Solver
def beam_search(start, goal, beam_width=3):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves
    
    start_time = time.time()
    beam = [(start, [start])]
    visited = set([serialize(start)])
    
    while beam:
        new_beam = []
        for state, path in beam:
            if state == goal:
                elapsed_time = time.time() - start_time
                return path, len(path) - 1, elapsed_time, True
            
            neighbors = get_neighbors(state)
            for neighbor in neighbors:
                ser = serialize(neighbor)
                if ser not in visited:
                    visited.add(ser)
                    new_beam.append((neighbor, path + [neighbor]))
        
        new_beam.sort(key=lambda x: manhattan_heuristic(x[0], goal))
        beam = new_beam[:beam_width]
        
        if not beam:
            break
    
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

# AND-OR Tree Search Solver
def and_or_tree_search(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def get_neighbors(state):
        moves = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    x, y = i, j
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                new_state = copy.deepcopy(state)
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                moves.append(new_state)
        return moves
    
    def or_search(state, path, visited):
        if state == goal:
            return path + [state]
        
        ser = serialize(state)
        if ser in visited:
            return None
        visited.add(ser)
        
        neighbors = get_neighbors(state)
        for neighbor in neighbors:
            result = or_search(neighbor, path + [state], visited)
            if result is not None:
                return result
        
        return None
    
    start_time = time.time()
    visited = set()
    result = or_search(start, [], visited)
    
    if result:
        elapsed_time = time.time() - start_time
        return result, len(result) - 1, elapsed_time, True
    
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

# CSP Helper Functions
def state_to_assignment(state):
    assignment = {}
    for i in range(3):
        for j in range(3):
            tile = state[i][j]
            assignment[tile] = (i, j)
    return assignment

def assignment_to_state(assignment):
    state = [[0 for _ in range(3)] for _ in range(3)]
    for tile, (i, j) in assignment.items():
        state[i][j] = tile
    return state

def is_valid_move(current_pos, new_pos):
    x1, y1 = current_pos
    x2, y2 = new_pos
    return abs(x1 - x2) + abs(y1 - y2) == 1

# Backtracking Search Solver (Modified)
def backtracking_search(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def is_valid_assignment(state, row, col, num):
        # Kiểm tra xem số num đã được sử dụng chưa
        flat = [state[i][j] for i in range(3) for j in range(3) if state[i][j] is not None]
        if num in flat:
            return False
        
        # Gán số tạm thời để kiểm tra ràng buộc
        state[row][col] = num
        
        # Kiểm tra ràng buộc: phần tử cuối cùng phải là 0
        if row == 2 and col == 2 and num != 0:
            state[row][col] = None
            return False
        
        # Kiểm tra ràng buộc: phần tử bên phải lớn hơn bên trái 1 đơn vị
        if col > 0 and state[row][col-1] is not None:
            if state[row][col] != state[row][col-1] + 1:
                state[row][col] = None
                return False
        
        # Kiểm tra ràng buộc: phần tử bên trái (nếu gán số cho cột > 0)
        if col < 2 and state[row][col+1] is not None:
            if state[row][col+1] != state[row][col] + 1:
                state[row][col] = None
                return False
        
        # Kiểm tra ràng buộc: phần tử bên dưới lớn hơn bên trên 3 đơn vị
        if row > 0 and state[row-1][col] is not None:
            if state[row][col] != state[row-1][col] + 3:
                state[row][col] = None
                return False
        
        # Kiểm tra ràng buộc: phần tử bên trên (nếu gán số cho hàng > 0)
        if row < 2 and state[row+1][col] is not None:
            if state[row+1][col] != state[row][col] + 3:
                state[row][col] = None
                return False
        
        state[row][col] = None
        return True
    
    def backtrack(state, pos, visited, steps):
        if pos == 9:  # Đã gán hết 9 ô
            if state[2][2] == 0:
                # Trả về trạng thái cuối cùng như một đường dẫn
                return [copy.deepcopy(state)], steps
            return None, steps
        
        row, col = divmod(pos, 3)
        if state[row][col] is not None:
            return backtrack(state, pos + 1, visited, steps)
        
        for num in range(9):  # Thử các số từ 0 đến 8
            if is_valid_assignment(state, row, col, num):
                state[row][col] = num
                ser = serialize(state)
                if ser not in visited:
                    visited.add(ser)
                    # Tiếp tục gán ô tiếp theo
                    result, new_steps = backtrack(state, pos + 1, visited, steps + 1)
                    if result is not None:
                        # Nếu tìm được đường dẫn đúng, thêm trạng thái hiện tại vào đầu đường dẫn
                        return [copy.deepcopy(state)] + result, new_steps
                state[row][col] = None  # Quay lại nếu gán sai
        
        return None, steps
    
    start_time = time.time()
    # Khởi tạo trạng thái ban đầu với None (chưa gán số)
    initial_state = [[None for _ in range(3)] for _ in range(3)]
    visited = set()
    result, total_steps = backtrack(initial_state, 0, visited, 0)
    
    if result:
        elapsed_time = time.time() - start_time
        return result, total_steps, elapsed_time, True
    
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

# Forward Checking Solver (Modified)
def forward_checking(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def forward_check(var, pos, assignment, domains):
        new_domains = copy.deepcopy(domains)
        for v in variables:
            if v not in assignment and pos in new_domains[v]:
                new_domains[v].remove(pos)
                if not new_domains[v]:
                    return None
        return new_domains
    
    def backtrack(assignment, variables, domains, path, visited):
        if len(assignment) == len(variables):
            state = assignment_to_state(assignment)
            if state == goal:
                return path + [state]
            return None
        
        var = variables[len(assignment)]
        for pos in domains[var]:
            if pos not in [assignment[v] for v in assignment]:
                if var == 0 or (var in assignment and is_valid_move(assignment[0], pos)):
                    assignment[var] = pos
                    state = assignment_to_state(assignment)
                    if serialize(state) not in visited:
                        visited.add(serialize(state))
                        new_domains = forward_check(var, pos, assignment, domains)
                        if new_domains is not None:
                            # Only append state to path for solution steps
                            result = backtrack(assignment, variables, new_domains, path + [state], visited)
                            if result is not None:
                                return result
                    del assignment[var]
        return None
    
    start_time = time.time()
    variables = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    domains = {v: [(i, j) for i in range(3) for j in range(3)] for v in variables}
    assignment = state_to_assignment(start)
    visited = set([serialize(start)])
    path = [start]
    
    result = backtrack(assignment, variables, domains, path, visited)
    if result:
        elapsed_time = time.time() - start_time
        return result, len(result) - 1, elapsed_time, True
    
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

# AC-3 Solver (Modified)
def ac3(start, goal):
    def serialize(state):
        return tuple(tuple(row) for row in state)
    
    def make_arc_consistent(domains):
        queue = [(v1, v2) for v1 in variables for v2 in variables if v1 != v2]
        while queue:
            v1, v2 = queue.pop(0)
            removed = False
            for x in domains[v1][:]:
                if not any(is_valid_move(x, y) if v1 == 0 or v2 == 0 else True for y in domains[v2]):
                    domains[v1].remove(x)
                    removed = True
                if not domains[v1]:
                    return None
                if removed:
                    for v3 in variables:
                        if v3 != v1 and v3 != v2:
                            queue.append((v3, v1))
        return domains
    
    def backtrack(assignment, variables, domains, path, visited):
        if len(assignment) == len(variables):
            state = assignment_to_state(assignment)
            if state == goal:
                return path + [state]
            return None
        
        var = variables[len(assignment)]
        for pos in domains[var]:
            if pos not in [assignment[v] for v in assignment]:
                if var == 0 or (var in assignment and is_valid_move(assignment[0], pos)):
                    assignment[var] = pos
                    state = assignment_to_state(assignment)
                    if serialize(state) not in visited:
                        visited.add(serialize(state))
                        # Only append state to path for solution steps
                        result = backtrack(assignment, variables, domains, path + [state], visited)
                        if result is not None:
                            return result
                    del assignment[var]
        return None
    
    start_time = time.time()
    variables = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    domains = {v: [(i, j) for i in range(3) for j in range(3)] for v in variables}
    
    domains = make_arc_consistent(domains)
    if domains is None:
        elapsed_time = time.time() - start_time
        return [], 0, elapsed_time, False
    
    assignment = state_to_assignment(start)
    visited = set([serialize(start)])
    path = [start]
    
    result = backtrack(assignment, variables, domains, path, visited)
    if result:
        elapsed_time = time.time() - start_time
        return result, len(result) - 1, elapsed_time, True
    
    elapsed_time = time.time() - start_time
    return [], 0, elapsed_time, False

# Game logic
class Puzzle:
    def __init__(self):
        self.state = [
                        [2, 4, 3],
                        [1, 5, 0],
                        [7, 8, 6]
                    ]
        self.start_state = [
                                [2, 4, 3],
                                [1, 5, 0],
                                [7, 8, 6]
                            ]
        self.goal_state = [[1, 2, 3], 
                           [4, 5, 6], 
                           [7, 8, 0]]
        self.solution = []
        self.step = 0
        self.scroll_offset = 0
        self.total_steps = 0
        self.elapsed_time = 0
        self.solved = False
        self.selected_algo = "BFS"
        self.algorithms = [
            "BFS", "DFS", "UCS", "IDS", "Greedy", "A*", "IDA*",
            "Hill Climbing", "Steepest-Ascent", "Stochastic HC",
            "Simulated Annealing", "Genetic Algorithm", "Beam Search",
            "Backtracking", "Forward Checking", "AC-3", "AND-OR Search"
        ]
        self.dropdown_open = False
        self.dropdown_scroll_offset = 0  # Biến mới để theo dõi cuộn dropdown
        self.max_dropdown_items = 5

    def draw_grid(self, base_x, base_y, cell_size, state):
        for i in range(3):
            for j in range(3):
                val = state[i][j]
                rect = pygame.Rect(base_x + j * cell_size, base_y + i * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, GRAY, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)
                if val != 0:
                    text = pygame.font.SysFont(None, 24).render(str(val), True, BLACK)
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

    def draw_dropdown(self):
        label_font = pygame.font.SysFont(None, 24)
        pygame.draw.rect(screen, BLUE, (600, 40, 120, 30))
        algo_text = label_font.render(self.selected_algo, True, WHITE)
        screen.blit(algo_text, (610, 45))
        self.dropdown_rect = pygame.Rect(600, 40, 120, 30)

        if self.dropdown_open:
            # Tính chỉ số bắt đầu và kết thúc dựa trên offset
            start_idx = self.dropdown_scroll_offset
            end_idx = min(start_idx + self.max_dropdown_items, len(self.algorithms))
            
            # Vẽ các mục trong phạm vi hiển thị
            for i, algo in enumerate(self.algorithms[start_idx:end_idx]):
                item_rect = pygame.Rect(600, 70 + i * 30, 120, 30)
                pygame.draw.rect(screen, BLUE, item_rect)
                text = label_font.render(algo, True, WHITE)
                screen.blit(text, (610, 75 + i * 30))
                # Lưu rect của mục để xử lý nhấp chuột
                if not hasattr(self, 'dropdown_item_rects'):
                    self.dropdown_item_rects = []
                if len(self.dropdown_item_rects) <= i:
                    self.dropdown_item_rects.append(item_rect)
                else:
                    self.dropdown_item_rects[i] = item_rect

    def draw_buttons(self):
        label_font = pygame.font.SysFont(None, 24)
        self.solve_button = pygame.Rect(400, 160 , 120, 30)
        self.reset_button = pygame.Rect(400, 200, 120, 30)
        pygame.draw.rect(screen, BLUE, self.solve_button)
        pygame.draw.rect(screen, BLUE, self.reset_button)
        screen.blit(label_font.render("Solve", True, WHITE), (435, 165))
        screen.blit(label_font.render("Reset", True, WHITE), (435, 205))

    def draw(self):
        screen.fill(WHITE)

        label_font = pygame.font.SysFont(None, 24)
        screen.blit(label_font.render("Start", True, BLACK), (50, 20))
        screen.blit(label_font.render("Goal", True, BLACK), (250, 20))
        self.draw_grid(50, 40, 30, self.start_state)
        self.draw_grid(250, 40, 30, self.goal_state)

        status_x = 400
        status_y = 40
        status_lines = [
            f"Steps: {self.total_steps}",
            f"Time: {self.elapsed_time:.3f}s",
            f"Solved: {'Yes' if self.solved else 'No'}"
        ]
        for idx, line in enumerate(status_lines):
            label = label_font.render(line, True, BLACK)
            screen.blit(label, (status_x, status_y + idx * 30))

        for i in range(3):
            for j in range(3):
                val = self.state[i][j]
                rect = pygame.Rect(j * GRID_SIZE + 50, i * GRID_SIZE + 160, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(screen, GRAY, rect)
                pygame.draw.rect(screen, BLACK, rect, 2)
                if val != 0:
                    text = FONT.render(str(val), True, BLACK)
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

        self.draw_dropdown()
        self.draw_buttons()

        pygame.draw.rect(screen, GRAY, (50, 500, 700, 200))
        pygame.draw.rect(screen, BLACK, (50, 500, 700, 200), 2)
        label = label_font.render("Solution Steps:", True, BLACK)
        screen.blit(label, (55, 505))

        for idx, step in enumerate(self.solution):
            y = 530 + idx * 75 - self.scroll_offset
            if 510 < y < 680:
                for i, row in enumerate(step):
                    row_text = ' '.join(str(num) if num is not None else '-' for num in row)
                    step_text = label_font.render(f"{idx if i == 0 else ' '}: {row_text}", True, BLACK)
                    screen.blit(step_text, (60, y + i * 20))

    def handle_click(self, pos):
        if self.solve_button.collidepoint(pos):
            self.solution = []
            self.step = 0
            self.scroll_offset = 0
            self.total_steps = 0
            self.elapsed_time = 0
            self.solved = False

            # Run the selected algorithm
            if self.selected_algo == "BFS":
                self.solution, self.total_steps, self.elapsed_time, self.solved = bfs(self.state)
            elif self.selected_algo == "DFS":
                self.solution, self.total_steps, self.elapsed_time, self.solved = dfs(self.state)
            elif self.selected_algo == "UCS":
                self.solution, self.total_steps, self.elapsed_time, self.solved = ucs(self.state)
            elif self.selected_algo == "IDS":
                self.solution, self.total_steps, self.elapsed_time, self.solved = ids(self.state)
            elif self.selected_algo == "Greedy":
                self.solution, self.total_steps, self.elapsed_time, self.solved = greedy(self.state, self.goal_state)
            elif self.selected_algo == "A*":
                self.solution, self.total_steps, self.elapsed_time, self.solved = a_star(self.state, self.goal_state)
            elif self.selected_algo == "IDA*":
                self.solution, self.total_steps, self.elapsed_time, self.solved = ida_star(self.state, self.goal_state)
            elif self.selected_algo == "Hill Climbing":
                self.solution, self.total_steps, self.elapsed_time, self.solved = hill_climbing(self.state, self.goal_state)
            elif self.selected_algo == "Steepest-Ascent":
                self.solution, self.total_steps, self.elapsed_time, self.solved = steepest_ascent_hill_climbing(self.state, self.goal_state)
            elif self.selected_algo == "Stochastic HC":
                self.solution, self.total_steps, self.elapsed_time, self.solved = stochastic_hill_climbing(self.state, self.goal_state)
            elif self.selected_algo == "Simulated Annealing":
                self.solution, self.total_steps, self.elapsed_time, self.solved = simulated_annealing(self.state, self.goal_state)
            elif self.selected_algo == "Genetic Algorithm":
                self.solution, self.total_steps, self.elapsed_time, self.solved = genetic_algorithm(self.state, self.goal_state)
            elif self.selected_algo == "Beam Search":
                self.solution, self.total_steps, self.elapsed_time, self.solved = beam_search(self.state, self.goal_state)
            elif self.selected_algo == "Backtracking":
                self.solution, self.total_steps, self.elapsed_time, self.solved = backtracking_search(self.state, self.goal_state)
            elif self.selected_algo == "Forward Checking":
                self.solution, self.total_steps, self.elapsed_time, self.solved = forward_checking(self.state, self.goal_state)
            elif self.selected_algo == "AC-3":
                self.solution, self.total_steps, self.elapsed_time, self.solved = ac3(self.state, self.goal_state)
            elif self.selected_algo == "AND-OR Search":
                self.solution, self.total_steps, self.elapsed_time, self.solved = and_or_tree_search(self.state, self.goal_state)

        elif self.reset_button.collidepoint(pos):
            self.state = copy.deepcopy(self.start_state)
            self.solution = []
            self.step = 0
            self.scroll_offset = 0
            self.total_steps = 0
            self.elapsed_time = 0
            self.solved = False
        elif self.dropdown_rect.collidepoint(pos):
            self.dropdown_open = not self.dropdown_open
            self.dropdown_item_rects = []
        elif self.dropdown_open:
            for i, item_rect in enumerate(self.dropdown_item_rects):
                if item_rect.collidepoint(pos):
                    # Tính chỉ số thực của thuật toán dựa trên offset
                    algo_idx = i + self.dropdown_scroll_offset
                    if algo_idx < len(self.algorithms):
                        self.selected_algo = self.algorithms[algo_idx]
                        self.dropdown_open = False
                        self.dropdown_item_rects = []
                        break

    def handle_scroll(self, direction):
        if self.dropdown_open:
            # Xử lý cuộn trong dropdown
            if direction == 'up':
                self.dropdown_scroll_offset = max(self.dropdown_scroll_offset - 1, 0)
            elif direction == 'down':
                self.dropdown_scroll_offset = min(self.dropdown_scroll_offset + 1, len(self.algorithms) - self.max_dropdown_items)
        else:
            # Xử lý cuộn trong solution steps (giữ nguyên)
            if direction == 'up':
                self.scroll_offset = max(self.scroll_offset - 30, 0)
            elif direction == 'down':
                self.scroll_offset += 30

    def update(self):
        # Only update the main grid for non-CSP algorithms
        if self.solution and self.step < len(self.solution):
            if self.selected_algo not in ["Backtracking", "Forward Checking", "AC-3"]:
                pygame.time.wait(300)
                self.state = self.solution[self.step]
                self.step += 1

def main():
    clock = pygame.time.Clock()
    puzzle = Puzzle()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    puzzle.handle_click(pygame.mouse.get_pos())
                elif event.button == 4:
                    puzzle.handle_scroll('up')
                elif event.button == 5:
                    puzzle.handle_scroll('down')

        puzzle.update()
        puzzle.draw()
        pygame.display.flip()
        clock.tick(30)

if __name__ == '__main__':
    main()