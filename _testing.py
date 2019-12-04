import time
import pickle
import copy
import random
import curses
import random
import operator
import numpy
import csv
from functools import partial
import pygraphviz as pgv

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

S_RIGHT, S_LEFT, S_UP, S_DOWN = 0, 1, 2, 3
XSIZE, YSIZE = 14, 14
BOARD_SIZE = (YSIZE - 2) * (XSIZE - 2)
NFOOD = 1
# Mutation rate for eaMuPlusLambda (current implementation MUPB = 1.0)
MUTPB = 0.8
CXPB = 0.1  # Crossover rate
NGEN = 300  # Number of generations
MU = 800  # Population size
LAMBDA = MU * 2  # Used in the unused eaMuPlusLambda


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)


def if_then_else(condition, out1, out2):
    out1() if condition() else out2()


class SnakePlayer(list):
    global S_RIGHT, S_LEFT, S_UP, S_DOWN
    global XSIZE, YSIZE

    def __init__(self):
        self.direction = S_RIGHT
        self.body = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6],
                     [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0], ]
        self.score = 0
        self.ahead = []
        self.food = []
        self.routine = None

    def _reset(self):
        self.direction = S_RIGHT
        self.body = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6],
                     [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0], ]
        self.score = 0
        self.ahead = []
        self.food = []

    def getAheadLocation(self):
        self.ahead = [
            self.body[0][0]
            + (self.direction == S_DOWN and 1)
            + (self.direction == S_UP and -1),
            self.body[0][1]
            + (self.direction == S_LEFT and -1)
            + (self.direction == S_RIGHT and 1),
        ]

    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead)

    # Movement functions
    def changeDirectionUp(self):
        self.direction = S_UP

    def changeDirectionRight(self):
        self.direction = S_RIGHT

    def changeDirectionDown(self):
        self.direction = S_DOWN

    def changeDirectionLeft(self):
        self.direction = S_LEFT

    def changeRelativeDirectionLeft(self):
        if self.direction == S_UP:
            self.changeDirectionLeft()
        elif self.direction == S_LEFT:
            self.changeDirectionDown()
        elif self.direction == S_DOWN:
            self.changeDirectionRight()
        elif self.direction == S_RIGHT:
            self.changeDirectionUp()

    def changeRelativeDirectionRight(self):
        if self.direction == S_UP:
            self.changeDirectionRight()
        elif self.direction == S_LEFT:
            self.changeDirectionUp()
        elif self.direction == S_DOWN:
            self.changeDirectionLeft()
        elif self.direction == S_RIGHT:
            self.changeDirectionDown()

    def moveForward(self):
        pass

    def get_head(self):
        return [self.body[0][0], self.body[0][1]]

    def snakeHasCollided(self):
        self.hit = False
        if (
            self.body[0][0] == 0
            or self.body[0][0] == (YSIZE - 1)
            or self.body[0][1] == 0
            or self.body[0][1] == (XSIZE - 1)
        ):
            self.hit = True
        if self.body[0] in self.body[1:]:
            self.hit = True
        return self.hit

    # Location functions
    def get_right(self, right_coord):
        if (self.direction == S_RIGHT):
            right_coord[0] += 1
        elif (self.direction == S_LEFT):
            right_coord[0] -= 1
        elif (self.direction == S_UP):
            right_coord[1] += 1
        else:
            right_coord[0] -= 1
        return right_coord

    def get_left(self, left_coord):
        if (self.direction == S_RIGHT):
            left_coord[0] -= 1
        elif (self.direction == S_LEFT):
            left_coord[0] += 1
        elif (self.direction == S_UP):
            left_coord[1] -= 1
        else:
            left_coord[0] += 1
        return left_coord

    def checkFoodDirection(self):
        head = self.get_head()
        food = self.food[0]
        dir_y = head[0] - food[0]
        dir_x = head[1] - food[1]
        return [dir_y, dir_x]

    # Sensing functions
    def sense_wall_ahead(self):
        self.getAheadLocation()
        return (
            self.ahead[0] == 0
            or self.ahead[0] == (YSIZE - 1)
            or self.ahead[1] == 0
            or self.ahead[1] == (XSIZE - 1)
        )

    def sense_food_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.food

    def sense_food_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.food

    def sense_tail_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.body

    def sense_obstacle_right(self):
        right_coord = self.get_right(self.get_head())
        return (right_coord in self.body) or checkWall(right_coord)

    def sense_obstacle_left(self):
        left_coord = self.get_left(self.get_head())
        return (left_coord in self.body) or checkWall(left_coord)
    
    def sense_food_is_right(self):
        coord = self.checkFoodDirection()
        return coord[1] < 0
    
    def sense_food_is_left(self):
        coord = self.checkFoodDirection()
        return coord[1] > 0
    
    def sense_food_is_up(self):
        coord = self.checkFoodDirection()
        return coord[0] > 0
    
    def sense_food_is_down(self):
        coord = self.checkFoodDirection()
        return coord[0] < 0


    # IF FUNCTIONS
    def if_food_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_food_ahead, out1, out2)

    def if_tail_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_tail_ahead, out1, out2)

    def if_obstacle_right(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_right, out1, out2)

    def if_obstacle_left(self, out1, out2):
        return partial(if_then_else, self.sense_obstacle_left, out1, out2)

    def if_food_is_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_is_right, out1, out2)

    def if_food_is_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_is_left, out1, out2)

    def if_food_is_up(self, out1, out2):
        return partial(if_then_else, self.sense_food_is_up, out1, out2)

    def if_food_is_down(self, out1, out2):
        return partial(if_then_else, self.sense_food_is_down, out1, out2)

def checkWall(coord):
    return (coord[0] == 0 
        or coord[0] == (YSIZE - 1) 
        or coord[1] == 0 
        or coord[1] == (XSIZE - 1))

# This function places a food item in the environment
def placeFood(snake):
    food = []
    while len(food) < NFOOD:
        potentialfood = [random.randint(
            1, (YSIZE - 2)), random.randint(1, (XSIZE - 2))]
        if not (potentialfood in snake.body) and not (potentialfood in food):
            food.append(potentialfood)
    snake.food = food  # let the snake know where the food is
    return food

def checkFood(snake):
	# For all the tiles on the board, check if they're free
    for y in range(0, YSIZE):  
        for x in range(0, XSIZE):
            coord = [y, x]
            if not (
                    coord in snake.body) and not (
                    coord in snake.food) and not (
                    checkWall(coord)):
                return True
    return False


snake = SnakePlayer()


def displayStrategyRun(individual):
    global snake
    global pset

    routine = gp.compile(individual, pset)

    curses.initscr()
    win = curses.newwin(YSIZE, XSIZE, 0, 0)
    win.keypad(1)
    curses.noecho()
    curses.curs_set(0)
    win.border(0)
    win.nodelay(1)
    win.timeout(120)

    snake._reset()
    food = placeFood(snake)

    for f in food:
        win.addch(f[0], f[1], "@")
    steps = 0
    timer = 0
    collided = False
    while not collided and not timer == ((2 * XSIZE) * YSIZE):
        # Set up the display
        win.border(0)
        win.addstr(0, 2, "Score : " + str(snake.score) + " ")
        win.getch()

        ## EXECUTE THE SNAKE'S BEHAVIOUR HERE ##
        routine()
        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            for f in food:
                win.addch(f[0], f[1], " ")
            food = placeFood(snake)
            for f in food:
                win.addch(f[0], f[1], "@")
            timer = 0
        else:
            last = snake.body.pop()
            win.addch(last[0], last[1], " ")
            timer += 1  # timesteps since last eaten
        win.addch(snake.body[0][0], snake.body[0][1], "o")

        collided = snake.snakeHasCollided()
        hitBounds = timer == ((2 * XSIZE) * YSIZE)
        steps += 1

    time.sleep(2)
    curses.endwin()

    print(collided)
    print(hitBounds)
    
    return (snake.score,)


def runGame(individual):
    global snake
    global pset

    routine = gp.compile(individual, pset)

    totalScore = 0
    steps = 0
    snake._reset()
    food = placeFood(snake)
    timer = 0
    while not snake.snakeHasCollided() and not timer == XSIZE * YSIZE:
        routine()
        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            if not (checkFood(snake)):
                print("Maximum score achieved!")
                return (snake.score,)  # If not return; CONGRATS ON MAX SCORE!!!
            else:  # If coords free, then place food
                food = placeFood(snake)
                timer = 0
        else:
            snake.body.pop()
            timer += 1  # timesteps since last eaten

        steps += 1

        totalScore += snake.score

    return (snake.score, )


def main(random_seed, checkpoint=False):
    global snake
    global pset

    random.seed(random_seed)

    pset = gp.PrimitiveSet("MAIN", 0)

    pset.addPrimitive(prog2, 2)
    pset.addPrimitive(prog3, 3)

    pset.addPrimitive(snake.if_food_ahead, 2)
    pset.addPrimitive(snake.if_tail_ahead, 2)
    pset.addPrimitive(snake.if_food_is_down, 2)
    pset.addPrimitive(snake.if_food_is_left, 2)
    pset.addPrimitive(snake.if_food_is_right, 2)
    pset.addPrimitive(snake.if_food_is_up, 2)
    pset.addPrimitive(snake.if_obstacle_left, 2)
    pset.addPrimitive(snake.if_obstacle_right, 2)

    pset.addTerminal(snake.changeRelativeDirectionRight)
    pset.addTerminal(snake.changeRelativeDirectionLeft)
    pset.addTerminal(snake.moveForward)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("expr_init", gp.genGrow, pset=pset, min_=1, max_=10)

    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evalScore(individual):
        return runGame(individual)

    def evalScoreVisual(individual):
        return displayStrategyRun(individual)

    toolbox.register("evaluate", evalScore)  # Evaluation method
    # Selection operator (generational)
    toolbox.register("select", tools.selNSGA2)
    # Selection operator (varying population)
    toolbox.register("select_sample", tools.selStochasticUniversalSampling)
    toolbox.register("mate", gp.cxOnePoint)  # Crossover operator
    toolbox.register("mutate", gp.mutInsert, pset=pset)  # Mutation operator

    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean, axis=0)
    mstats.register("std", numpy.std, axis=0)
    mstats.register("min", numpy.min, axis=0)
    mstats.register("max", numpy.max, axis=0)


    if checkpoint:
        # A file name has been given, then load the data from the file
        with open('checkpoint.pkl', "r") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
    else:
        # Start a new evolution
        population = toolbox.population(n=300)
        start_gen = 0
        hof = tools.HallOfFame(maxsize=1)
        logbook = tools.Logbook()

    pop, log = algorithms.eaSimple(population, toolbox, 0.5, 0.1, 50, stats=mstats,
                                halloffame=hof, verbose=True)

    cp = dict(population=population, halloffame=hof,
                      logbook=logbook, rndstate=random.getstate())

    with open("checkpoint.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)    

    epr = tools.selBest(pop, 1)[0]
    displayStrategyRun(epr)
    nodes, edges, labels = gp.graph(epr)

    g = pgv.AGraph(nodeSep=1.0)
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")

if __name__ == "__main__":
    main(35, checkpoint=False)