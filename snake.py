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
# import pygraphviz as pgv
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

S_RIGHT, S_LEFT, S_UP, S_DOWN = 0, 1, 2, 3
XSIZE, YSIZE = 14, 14
# NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)
NFOOD = 1


def progn(*args):
    for arg in args:
        arg()


def prog2(out1, out2):
    return partial(progn, out1, out2)


def prog3(out1, out2, out3):
    return partial(progn, out1, out2, out3)


def if_then_else(condition, out1, out2):
    out1() if condition() else out2()


# This class can be used to create a basic player object (snake agent)
class SnakePlayer(list):
    global S_RIGHT, S_LEFT, S_UP, S_DOWN
    global XSIZE, YSIZE

    def __init__(self):
        self.direction = S_RIGHT
        self.body = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6],
                     [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.score = 0
        self.ahead = []
        self.food = []

    def _reset(self):
        self.direction = S_RIGHT
        self.body[:] = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6],
                        [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.score = 0
        self.ahead = []
        self.food = []

    def getAheadLocation(self):
        self.ahead = [self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1),
                      self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)]

    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead)

    # You are free to define more sensing options to the snake

    def changeDirectionUp(self):
        self.direction = S_UP

    def changeDirectionRight(self):
        self.direction = S_RIGHT

    def changeDirectionDown(self):
        self.direction = S_DOWN

    def changeDirectionLeft(self):
        self.direction = S_LEFT

    def snakeHasCollided(self):
        self.hit = False
        if self.body[0][0] == 0 or self.body[0][0] == (
                YSIZE-1) or self.body[0][1] == 0 or self.body[0][1] == (XSIZE-1):
            self.hit = True
        if self.body[0] in self.body[1:]:
            self.hit = True
        return(self.hit)

    def get_up(self):
        return [self.body[0][0]-1, self.body[0][1]]

    def get_down(self):
        return [self.body[0][0]+1, self.body[0][1]]

    def get_left(self):
        return [self.body[0][0], self.body[0][1]-1]

    def get_right(self):
        return [self.body[0][0], self.body[0][1]+1]

    # Sense wall ahead
    def sense_object_ahead(self):
        self.getAheadLocation()
        return (self.ahead[0] == 0 or self.ahead[0] == (YSIZE-1) or
               self.ahead[1] == 0 or self.ahead[1] == (XSIZE-1) or
               self.ahead in self.body)

    def if_object_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_object_ahead, out1, out2)

    # Sense food ahead
    def sense_food_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.food

    def if_food_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_food_ahead, out1, out2)

    # Sense if food is to the left
    def sense_food_left(self):
        return self.food[0][1] < self.body[0][1]

    def if_food_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_left, out1, out2)

    # Sense if food is to the right
    def sense_food_right(self):
        return self.food[0][1] > self.body[0][1]

    def if_food_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_right, out1, out2)

    # Sense if food is to the down
    def sense_food_down(self):
        return self.food[0][0] > self.body[0][0]

    def if_food_down(self, out1, out2):
        return partial(if_then_else, self.sense_food_down, out1, out2)

    # Sense if food is to the up
    def sense_food_up(self):
        return self.food[0][0] < self.body[0][0]

    def if_food_up(self, out1, out2):
        return partial(if_then_else, self.sense_food_up, out1, out2)

    # Sense objects in all directions
    def sense_object_up(self):
        coord = self.get_up()
        return coord in self.body or checkWall(coord)

    def if_object_up(self, out1, out2):
        return partial(if_then_else, self.sense_object_up, out1, out2)

    def sense_object_down(self):
        coord = self.get_down()
        return coord in self.body or checkWall(coord)

    def if_object_down(self, out1, out2):
        return partial(if_then_else, self.sense_object_down, out1, out2)

    def sense_object_left(self):
        coord = self.get_left()
        return coord in self.body or checkWall(coord)

    def if_object_left(self, out1, out2):
        return partial(if_then_else, self.sense_object_left, out1, out2)

    def sense_object_right(self):
        coord = self.get_right()
        return coord in self.body or checkWall(coord)

    def if_object_right(self, out1, out2):
        return partial(if_then_else, self.sense_object_right, out1, out2)

    # Sense corners
    def if_in_corner_left_up(self, out1, out2):
        return partial(if_then_else, self.sense_object_left and self.sense_object_up, out1, out2)

    def if_in_corner_left_down(self, out1, out2):
        return partial(if_then_else, self.sense_object_left and self.sense_object_down, out1, out2)

    def if_in_corner_right_up(self, out1, out2):
        return partial(if_then_else, self.sense_object_right and self.sense_object_up, out1, out2)

    def if_in_corner_right_down(self, out1, out2):
        return partial(if_then_else, self.sense_object_right and self.sense_object_down, out1, out2)

    # Sense Size
    def sense_body_longer_than_2X(self):
        return len(self.body) > XSIZE*2

    def if_body_longer_than_2X(self, out1, out2):
        return partial(if_then_else, self.sense_body_longer_than_2X, out1, out2)
 
    def sense_body_longer_than_2Y(self):
        return len(self.body) > YSIZE*2

    def if_body_longer_than_2Y(self, out1, out2):
        return partial(if_then_else, self.sense_body_longer_than_2Y, out1, out2)

    # Sense direct food direction
    def sense_food_on_x_axis_up(self):
        return self.food[0][1] == self.body[0][1] and self.food[0][0] < self.body[0][0]

    def if_food_on_x_axis_up(self, out1, out2):
        return partial(if_then_else, self.sense_food_on_x_axis_up, out1, out2)

    def sense_food_on_x_axis_down(self):
        return self.food[0][1] == self.body[0][1] and self.food[0][0] > self.body[0][0]

    def if_food_on_x_axis_down(self, out1, out2):
        return partial(if_then_else, self.sense_food_on_x_axis_down, out1, out2)

    def sense_food_on_y_axis_left(self):
        return self.food[0][0] == self.body[0][0] and self.food[0][1] < self.body[0][1]

    def if_food_on_y_axis_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_on_y_axis_left, out1, out2)

    def sense_food_on_y_axis_right(self):
        return self.food[0][0] == self.body[0][0] and self.food[0][1] > self.body[0][1]

    def if_food_on_y_axis_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_on_y_axis_right, out1, out2)


def checkWall(coord):
    return (coord[0] == 0
            or coord[0] == (YSIZE - 1)
            or coord[1] == 0
            or coord[1] == (XSIZE - 1))


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


def placeFood(snake):
    food = []
    while len(food) < NFOOD:
        potentialfood = [random.randint(
            1, (YSIZE-2)), random.randint(1, (XSIZE-2))]
        if not (potentialfood in snake.body) and not (potentialfood in food):
            food.append(potentialfood)
    snake.food = food  # let the snake know where the food is
    return(food)


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
    win.timeout(10)

    snake._reset()
    food = placeFood(snake)

    for f in food:
        win.addch(f[0], f[1], '@')

    timer = 0
    collided = False
    while not collided and not timer == ((2*XSIZE) * YSIZE):

        # Set up the display
        win.border(0)
        win.addstr(0, 2, 'Score : ' + str(snake.score) + ' ')
        win.getch()

        routine()
        snake.updatePosition()

        if snake.body[0] in food:
            snake.score += 1
            for f in food:
                win.addch(f[0], f[1], ' ')
            food = placeFood(snake)
            for f in food:
                win.addch(f[0], f[1], '@')
            timer = 0
        else:
            last = snake.body.pop()
            win.addch(last[0], last[1], ' ')
            timer += 1  # timesteps since last eaten
        win.addch(snake.body[0][0], snake.body[0][1], 'o')

        collided = snake.snakeHasCollided()
        hitBounds = (timer == ((2*XSIZE) * YSIZE))

    time.sleep(1)
    curses.endwin()

    print("Collided: {}".format(collided))
    print("Bounds hit: {}".format(hitBounds))
    return snake.score,


def runGame(individual):
    global snake

    totalScore = 0
    routine = gp.compile(individual, pset)
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
                # If not return; CONGRATS ON MAX SCORE!!!
                return (snake.score,)
            else:  # If coords free, then place food
                food = placeFood(snake)
                timer = 0
        else:
            snake.body.pop()
            timer += 1  # timesteps since last eaten

    totalScore += snake.score
    return totalScore,


def runGameAvg(individual):
    global avg_scores
    avgs = [runGame(individual)[0] for x in range(4)]
    avg = sum(avgs)/len(avgs)
    avg_scores.append(avg)
    return (avg,)


def main(rseed=300):
    global snake
    global pset
    global avg_scores
    avg_scores = []
    random.seed(rseed)

    pset = gp.PrimitiveSet("MAIN", 0)

    pset.addPrimitive(prog2, 2)
    pset.addPrimitive(prog3, 3)

    # pset.addPrimitive(snake.if_food_ahead, 2)
    # pset.addPrimitive(snake.if_object_ahead, 2)

    pset.addPrimitive(snake.if_food_right, 2)
    pset.addPrimitive(snake.if_food_down, 2)
    pset.addPrimitive(snake.if_food_left, 2)
    pset.addPrimitive(snake.if_food_up, 2)

    pset.addPrimitive(snake.if_object_right, 2)
    pset.addPrimitive(snake.if_object_left, 2)
    pset.addPrimitive(snake.if_object_up, 2)
    pset.addPrimitive(snake.if_object_down, 2)

    pset.addPrimitive(snake.if_body_longer_than_2X, 2)
    pset.addPrimitive(snake.if_body_longer_than_2Y, 2)

    # pset.addPrimitive(snake.if_in_corner_left_up, 2)
    # pset.addPrimitive(snake.if_in_corner_left_down, 2)
    # pset.addPrimitive(snake.if_in_corner_right_up, 2)
    # pset.addPrimitive(snake.if_in_corner_right_down, 2)

    # pset.addPrimitive(snake.if_food_on_x_axis_up, 2)
    # pset.addPrimitive(snake.if_food_on_x_axis_down, 2)
    # pset.addPrimitive(snake.if_food_on_y_axis_right, 2)
    # pset.addPrimitive(snake.if_food_on_y_axis_left, 2)

    pset.addTerminal(snake.changeDirectionDown)
    pset.addTerminal(snake.changeDirectionLeft)
    pset.addTerminal(snake.changeDirectionRight)
    pset.addTerminal(snake.changeDirectionUp)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree,
                   fitness=creator.FitnessMax, pset=pset)
    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate,
                     creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", runGameAvg)
    toolbox.register("select", tools.selTournament, tournsize=7)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=20))

    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", numpy.mean, axis=0)
    mstats.register("std", numpy.std, axis=0)
    mstats.register("min", numpy.min, axis=0)
    mstats.register("max", numpy.max, axis=0)

    # Start a new evolution
    population = toolbox.population(n=200)
    hof = tools.HallOfFame(maxsize=1)
    pop, log = algorithms.eaSimple(population, toolbox, 0.5, 0.1, 120,
                                   stats=mstats, halloffame=hof,
                                   verbose=True)
    smoothing_factor = 20
    avg_scores = [sum(avg_scores[x:x+smoothing_factor])/smoothing_factor for x in range(0, len(avg_scores), smoothing_factor)]
    plt.plot(list(range(len(avg_scores))), avg_scores)
    plt.show()

    epr = tools.selBest(hof, 1)[0]
    # displayStrategyRun(epr)
    best_run_5_times = [runGame(epr)[0] for x in range(5)]
    print("Best from pop, run 5 times: {}".format(best_run_5_times))
    print("Best from pop, avg: {}".format(sum(best_run_5_times)/5))

    # nodes, edges, labels = gp.graph(epr)
    # g = pgv.AGraph(nodeSep=1.0)
    # g.add_nodes_from(nodes)
    # g.add_edges_from(edges)
    # g.layout(prog="dot")

    # for i in nodes:
    #     n = g.get_node(i)
    #     n.attr["label"] = labels[i]

    # g.draw("tree.pdf")

main()
