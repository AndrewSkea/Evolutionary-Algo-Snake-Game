import time
import sys
import os.path
import copy
import random
import curses
import random
import operator
import numpy
import csv
from functools import partial
import multiprocessing
import time
import statistics
from itertools import chain
from variables import *

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


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

    def get_walls(self):
        return ([[0, x] for x in list(range(XSIZE))] +
                [[y, 0] for y in list(range(YSIZE))] +
                [[YSIZE, x] for x in list(range(XSIZE))] +
                [[y, XSIZE] for y in list(range(YSIZE))])

    def getAheadLocation(self):
        self.ahead = [self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1),
                      self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)]

    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead)

    # You are free to define more sensing options to the snake

    def keepSameDirection(self):
        pass

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

    # Helper functions
    def get_up(self, x=1):
        return [max(self.body[0][0]-x, 0), self.body[0][1]]

    def get_down(self, x=1):
        return [min(self.body[0][0]+x, YSIZE), self.body[0][1]]

    def get_left(self, x=1):
        return [self.body[0][0], max(self.body[0][1]-x, 0)]

    def get_right(self, x=1):
        return [self.body[0][0], min(self.body[0][1]+x, XSIZE)]

    def check_for_object(self, coord):
        return checkWall(coord) or coord in self.body

    range_ar = list(range(2, 4))

    ##########
    # Sense if object is close
    def sense_if_object_is_to_right(self):
        return any([self.check_for_object(self.get_right(x)) for x in self.range_ar])

    def if_object_is_to_right(self, out1, out2):
        return partial(if_then_else, self.sense_if_object_is_to_right, out1, out2)

    def sense_if_object_is_to_left(self):
        return any([self.check_for_object(self.get_left(x)) for x in self.range_ar])

    def if_object_is_to_left(self, out1, out2):
        return partial(if_then_else, self.sense_if_object_is_to_left, out1, out2)

    def sense_if_object_is_up(self):
        return any([self.check_for_object(self.get_up(x)) for x in self.range_ar])

    def if_object_is_up(self, out1, out2):
        return partial(if_then_else, self.sense_if_object_is_up, out1, out2)

    def sense_if_object_is_down(self):
        return any([self.check_for_object(self.get_down(x)) for x in self.range_ar])

    def if_object_is_down(self, out1, out2):
        return partial(if_then_else, self.sense_if_object_is_down, out1, out2)
    ##########

    ##########
    # Sense where food in respect to head
    def sense_food_left(self):
        return self.food[0][1] < self.body[0][1]

    def if_food_left(self, out1, out2):
        return partial(if_then_else, self.sense_food_left, out1, out2)

    def sense_food_right(self):
        return self.food[0][1] > self.body[0][1]

    def if_food_right(self, out1, out2):
        return partial(if_then_else, self.sense_food_right, out1, out2)

    def sense_food_down(self):
        return self.food[0][0] > self.body[0][0]

    def if_food_down(self, out1, out2):
        return partial(if_then_else, self.sense_food_down, out1, out2)

    def sense_food_up(self):
        return self.food[0][0] < self.body[0][0]

    def if_food_up(self, out1, out2):
        return partial(if_then_else, self.sense_food_up, out1, out2)
    ##########

    def sense_object_ahead(self):
        self.getAheadLocation()
        coord = self.ahead
        return coord in self.body or checkWall(coord)

    def if_object_ahead(self, out1, out2):
        return partial(if_then_else, self.sense_object_ahead, out1, out2)


    ##########
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

    ##########


    ##########
    # Sense corners
    def if_in_corner_left_up(self, out1, out2):
        return partial(if_then_else, self.sense_object_left and self.sense_object_up, out1, out2)

    def if_in_corner_left_down(self, out1, out2):
        return partial(if_then_else, self.sense_object_left and self.sense_object_down, out1, out2)

    def if_in_corner_right_up(self, out1, out2):
        return partial(if_then_else, self.sense_object_right and self.sense_object_up, out1, out2)

    def if_in_corner_right_down(self, out1, out2):
        return partial(if_then_else, self.sense_object_right and self.sense_object_down, out1, out2)
    ##########


    ##########
    # Sense Size
    def sense_body_longer_than_X_by(self, multiplier=2):
        return len(self.body) >= XSIZE*multiplier

    def if_body_longer_than_2X(self, out1, out2):
        return partial(if_then_else, self.sense_body_longer_than_X_by, out1, out2)
 
    def sense_body_longer_than_Y_by(self, multiplier=2):
        return len(self.body) >= YSIZE*multiplier

    def if_body_longer_than_2Y(self, out1, out2):
        return partial(if_then_else, self.sense_body_longer_than_Y_by, out1, out2)
    ##########


    ##########
    # Sense if more space in one direction
    def sense_more_space_left(self):
        return True if self.body[0][1] <= XSIZE/2-1 else False

    def if_more_space_left(self, out1, out2):
        return partial(if_then_else, self.sense_more_space_left, out1, out2)

    def sense_more_space_up(self):
        return True if self.body[0][0] <= YSIZE/2-1 else False

    def if_more_space_up(self, out1, out2):
        return partial(if_then_else, self.sense_more_space_up, out1, out2)

    def sense_more_space_right(self):
        return True if self.body[0][1] > XSIZE/2-1 else False

    def if_more_space_right(self, out1, out2):
        return partial(if_then_else, self.sense_more_space_right, out1, out2)

    def sense_more_space_down(self):
        return True if self.body[0][0] > YSIZE/2-1 else False

    def if_more_space_down(self, out1, out2):
        return partial(if_then_else, self.sense_more_space_down, out1, out2)
    ##########


    ##########
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
    ##########


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
    win.timeout(20)
    steps = 0
    snake._reset()
    food = placeFood(snake)
    timer_array = []
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
        steps += 1
        if snake.body[0] in food:
            snake.score += 1
            for f in food:
                win.addch(f[0], f[1], ' ')
            food = placeFood(snake)
            for f in food:
                win.addch(f[0], f[1], '@')
            timer_array.append(timer)
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
    return snake.score, steps, timer_array


def runGame(individual):
    global snake

    totalScore = 0
    routine = gp.compile(individual, pset)
    snake._reset()
    food = placeFood(snake)
    timer_array = []
    timer = 0
    steps = 0
    while not snake.snakeHasCollided() and not timer == 2 * XSIZE * YSIZE:

        routine()
        snake.updatePosition()

        steps += 1
        if snake.body[0] in food:
            snake.score += 1
            if not (checkFood(snake)):
                print("Maximum score achieved!")
                # If not return; CONGRATS ON MAX SCORE!!!
                return (snake.score,)
            else:  # If coords free, then place food
                food = placeFood(snake)
                timer_array.append(timer)
                timer = 0
        else:
            snake.body.pop()
            timer += 1  # timesteps since last eaten

    final_distance_from_food = abs(snake.body[0][0] - food[0][0]) + abs(snake.body[0][1] - food[0][1])
    totalScore += snake.score
    return totalScore, steps, timer_array, final_distance_from_food


def runGameAvg(individual):
    global avg_scores
    index = 0
    index = 1 if FIT_FUNC == "steps" else 0
    index = 2 if FIT_FUNC == "timer" else 0
    index = 3 if FIT_FUNC == "dist" else 0
    averages = []

    for x in range(INDIVIDUAL_ITER):
        res = runGame(individual)
        averages.append(res[index])

    ret = sum(averages)/len(averages)
    ret = max(averages) if FIT_TYPE == "max" else ret
    ret = min(averages) if FIT_TYPE == "min" else ret
    return ret,


def main(rseed=300, use_last_best=False):
    global snake
    global pset
    global avg_scores
    avg_scores = []
    random.seed(rseed)

    pset = gp.PrimitiveSet("MAIN", 0)

    pset.addPrimitive(prog2, 2)
    pset.addPrimitive(prog3, 3)

    pset.addPrimitive(snake.if_object_is_to_right, 2)
    pset.addPrimitive(snake.if_object_is_to_left, 2)
    pset.addPrimitive(snake.if_object_is_up, 2)
    pset.addPrimitive(snake.if_object_is_down, 2)

    pset.addPrimitive(snake.if_food_right, 2)
    pset.addPrimitive(snake.if_food_down, 2)
    pset.addPrimitive(snake.if_food_left, 2)
    pset.addPrimitive(snake.if_food_up, 2)

    # pset.addPrimitive(snake.if_object_ahead, 2)

    # pset.addPrimitive(snake.if_body_longer_than_2X, 2)
    # pset.addPrimitive(snake.if_body_longer_than_2Y, 2)

    pset.addPrimitive(snake.if_object_right, 2)
    pset.addPrimitive(snake.if_object_left, 2)
    pset.addPrimitive(snake.if_object_up, 2)
    pset.addPrimitive(snake.if_object_down, 2)

    # pset.addTerminal(snake.keepSameDirection)
    pset.addTerminal(snake.changeDirectionDown)
    pset.addTerminal(snake.changeDirectionLeft)
    pset.addTerminal(snake.changeDirectionRight)
    pset.addTerminal(snake.changeDirectionUp)

    creator.create("FitnessMax", base.Fitness, weights=(1,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax, pset=pset)
    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", runGameAvg)

    # toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_FITNESS_SIZE)
    # toolbox.register("select", tools.selBest)
    kwargs = {"fitness_size": TOURNAMENT_FITNESS_SIZE, "parsimony_size": PARSIMONY_SIZE, "fitness_first": False}
    toolbox.register("select", tools.selDoubleTournament, **kwargs)

    kwargs = {"termpb": 0.1}
    toolbox.register("mate", gp.cxOnePointLeafBiased, **kwargs)
    # toolbox.register("mate", gp.cxOnePoint)

    # toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=CX_ONEPOINT_MAX_SIZE)
    toolbox.register("expr_mut", gp.genGrow, min_=1, max_=CX_ONEPOINT_MAX_SIZE)
    # toolbox.register("expr_mut", gp.genFull, min_=1, max_=CX_ONEPOINT_MAX_SIZE)

    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    # toolbox.register("mutate", gp.mutShrink)
    # toolbox.register("mutate", gp.mutNodeReplacement, pset=pset)
    # toolbox.register("mutate", gp.mutInsert, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT)) # For bloating
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=MAX_HEIGHT)) # For bloating

    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(fitness=stats_fit)
    mstats.register("avg", numpy.mean, axis=0)
    mstats.register("std", numpy.std, axis=0)
    mstats.register("min", numpy.min, axis=0)
    mstats.register("max", numpy.max, axis=0)

    population = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(maxsize=HOF_SIZE)

    # pop, log = algorithms.eaMuPlusLambda(population, toolbox, 25, 150, 0.5, 0.22, 120, stats=mstats, halloffame=hof, verbose=True)
    # pop, log = algorithms.eaMuCommaLambda(population, toolbox, 25, 275, 0.5, 0.2, 120, stats=mstats, halloffame=hof, verbose=True)
    pop, log = algorithms.eaSimple(population, toolbox, CROSS_PROB, MUT_PROB, ITERATIONS, stats=mstats, halloffame=hof, verbose=True)
    
    # smoothing_factor = 20
    # avg_scores = [sum(avg_scores[x:x+smoothing_factor])/smoothing_factor for x in range(0, len(avg_scores), smoothing_factor)]
    # plt.plot(list(range(len(avg_scores))), avg_scores)
    # plt.show()

    epr = tools.selBest(hof, 1)[0]
    iterations = 1000
    runs = [runGame(epr)[0] for x in range(iterations)]
    print("Best from pop, run 5 times: {}".format(runs))
    print("Best from pop, avg: {}".format(sum(runs)/len(runs)))
    return runs

# Section: Graphing
    # nodes, edges, labels = gp.graph(epr)
    # g = nx.Graph()
    # g.add_nodes_from(nodes)
    # g.add_edges_from(edges)
    # pos = nx.nx_agraph.graphviz_layout(g, prog="dot")

    # nx.draw_networkx_nodes(g, pos)
    # nx.draw_networkx_edges(g, pos)
    # nx.draw_networkx_labels(g, pos, labels)
    # plt.show()

    # nodes, edges, labels = gp.graph(epr)
    # g = pgv.AGraph(nodeSep=1.0)
    # g.add_nodes_from(nodes)
    # g.add_edges_from(edges)
    # g.layout(prog="dot")

    # for i in nodes:
    #     n = g.get_node(i)
    #     n.attr["label"] = labels[i]

    # g.draw("tree.pdf")
# Section End: Graphing


def parse_results(results, seeds):
    final_string = ""
    means = [statistics.mean(x) for x in results]
    medians = [statistics.median(x) for x in results]
    modes = [statistics.mode(x) for x in results]
    for i in range(len(results)):
        s = "{}\n\nMean:{}\nMedian:{}\nMode:{}\nSeed:{}\n\n\n".format(
            str(results[i]), means[i], medians[i], modes[i], seeds[i]
        )
        final_string += s

    final_string += "Seeds:\n{}\n\n".format(seeds)
    final_string += "Means:\n{}\n\n".format(means)
    final_string += "Medians:\n{}\n\n".format(medians)
    final_string += "Modes:\n{}\n\n".format(modes)

    results_flat = list(chain.from_iterable(results))
    final_string += "\nTop Level Results:\nMean:{}\nMedian:{}\nMode:{}\n\n\n\n".format(
        statistics.mean(results_flat), statistics.median(results_flat), statistics.mode(results_flat)
    )

    with open('./variables.py') as f:
        var_string = "\n\n\n\n\nVariables:\n{}".format(f.read())
        final_string += var_string

    return final_string


if __name__ == "__main__":

    name = ""
    if len(sys.argv) > 1:
        name = sys.argv[1] if sys.argv[1] != "" else ""
             

    timestr = time.strftime("%Y%m%d-%H%M%S")
    seeds = [random.randint(0, 1000) for i in range(30)]
    print(seeds)
    results = []
    for i in seeds:
        results.append(main(rseed=i, use_last_best=False))

    filename = "results/" + str(timestr) + name + ".txt"

    with open(filename, 'w') as f:
        f.write(parse_results(results, seeds))
