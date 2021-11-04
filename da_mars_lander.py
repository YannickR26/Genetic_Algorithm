from dataclasses import dataclass
import math
import sys
from random import randint, random
from typing import Optional
from copy import deepcopy
import numpy as np
from numpy.lib.function_base import average
import sdl2
import sdl2.ext
import time
import json
import pygal


MAP_WIDTH = 7000
MAP_HEIGHT = 3000
DIVISOR_FOR_PRINTER = 5

MIN_ROTATION_ANGLE = -90
MAX_ROTATION_ANGLE = 90
STEP_ROTATION_ANGLE = 15
MIN_POWER = 0
MAX_POWER = 4
STEP_POWER = 1
MAX_V_SPEED = 40  # Max vertical speed for landing
MAX_H_SPEED = 20  # Max horizontal speed for landing
MARS_GRAVITY = 3.711

CHROMOSOME_SIZE = 150
POPULATION_SIZE = 20

GRADED_RETAIN_PERCENT = 0.5
CHANCE_TO_MUTATE = 0.1
CHANCE_RETAIN_NONGRATED = 0.1

# Maximum number of generation before stopping the script
GENERATION_COUNT_MAX = 5


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


@dataclass
class Point:
    x: float
    y: float

    def __str__(self) -> str:
        return f"({round(self.x)}, {round(self.y)})"

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __ne__(self, other) -> bool:
        return self.x != other.x or self.y != other.y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    @staticmethod
    def distance_between(p1, p2) -> float:
        return round(math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2), 2)


class Line:
    def __init__(self, point0: Point = Point(0, 0), point1: Point = Point(0, 0)):
        self._point0: Point = point0
        self._point1: Point = point1
        self._landing_zone_direction: int = -1

    @property
    def point0(self) -> int:
        return self._point0

    @property
    def point1(self) -> int:
        return self._point1

    def line_to_point(self) -> tuple:
        return (round(self._point0.x//DIVISOR_FOR_PRINTER),
                round(MAP_HEIGHT//DIVISOR_FOR_PRINTER - self._point0.y//DIVISOR_FOR_PRINTER),
                round(self._point1.x//DIVISOR_FOR_PRINTER),
                round(MAP_HEIGHT//DIVISOR_FOR_PRINTER - self._point1.y//DIVISOR_FOR_PRINTER))

    @property
    def landing_zone_direction(self) -> int:
        return self._landing_zone_direction

    @landing_zone_direction.setter
    def landing_zone_direction(self, value: int):
        self._landing_zone_direction = value

    @property
    def length(self) -> float:
        return Point.distance_between(self._point0, self._point1)

    @property
    def mid_point(self) -> Point:
        return Point((self._point1.x - self._point0.x) / 2 + self._point0.x, (self._point1.y - self._point0.y) / 2 + self._point0.y)


class Surface:
    def __init__(self) -> None:
        self._lines: list[Line] = []
        self._landing_line_idx: int = -1

    @property
    def lines(self) -> list[Line]:
        return self._lines

    def get_max_distance(self) -> float:
        dist = 0.0
        for line in self._lines:
            dist += line.length
        return dist

    def create_surface_from_point(self, data: list):
        print(f"Create surface with data: {data}")
        point0 = Point(0, 0)
        landing_zone_found = False
        for i, point in enumerate(data):
            point1 = Point(int(point[0]), int(point[1]))
            if i != 0:
                line = Line(point0, point1)
                if line.point0.y == line.point1.y:
                    landing_zone_found = True
                    self._landing_line_idx = len(self._lines)
                    line.landing_zone_direction = 0
                elif landing_zone_found:
                    line.landing_zone_direction = 1
                self._lines.append(line)
            point0 = point1

    def collision_with_surface(self, point0: Point, point1: Point):
        if not 0 < point1.x < MAP_WIDTH or not 0 < point1.y < MAP_HEIGHT:
            return True, 0, False

        for idx, line in enumerate(self._lines):
            s1x = point1.x - point0.x
            s1y = point1.y - point0.y
            s2x = line.point1.x - line.point0.x
            s2y = line.point1.y - line.point0.y
            div = -s2x * s1y + s1x * s2y
            if div != 0:
                s = (
                    -s1y * (point0.x - line.point0.x) + s1x * (point0.y - line.point0.y)
                ) / (div)
                t = (
                    s2x * (point0.y - line.point0.y) - s2y * (point0.x - line.point0.x)
                ) / (div)
                if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
                    return True, idx, line.landing_zone_direction == 0

        return False, 0, False

    def find_distance_to_landing_zone(self, crash_point: Point, crash_line_idx: int) -> float:
        dist_to_landing = Point.distance_between(self._lines[self._landing_line_idx].mid_point, crash_point)
        return dist_to_landing


class Shuttle:
    def __init__(self) -> None:
        self._h_speed: float = 0.0  # The horizontal speed (in m/s), can be negative
        self._v_speed: float = 0.0  # The vertical speed (in m/s), can be negative
        self._rotate: int = 0  # The rotation angle in degrees (-90 to 90)
        self._power: int = 0  # The thrust power (0 to 4)
        self._fuel: int = 1000  # The quantity of remaining fuel in liters
        self._crashed: bool = False
        self._crash_on_landing_zone: bool = False
        self._crash_distance: float = 0.0  # Distance between crash and landing zone
        self._successfull_landing: bool = False
        self._trajectory: list[tuple[Point, int, int]] = []

    def __str__(self) -> str:
        return (
            f"pos: {self.position}, "
            + f"h_speed: {round(self._h_speed)}, "
            + f"v_speed: {round(self._v_speed)}, "
            + f"rotate: {self._rotate}, "
            + f"power: {self._power}, "
            + f"fuel: {self._fuel}"
        )

    @property
    def position(self) -> Point:
        return self._trajectory[-1][0]

    @position.setter
    def position(self, value: Point):
        self._trajectory.append((value, self._rotate, self._power))

    @property
    def v_speed(self) -> float:
        return self._v_speed

    @v_speed.setter
    def v_speed(self, value: float):
        self._v_speed = value

    @property
    def h_speed(self) -> float:
        return self._h_speed

    @h_speed.setter
    def h_speed(self, value: float):
        self._h_speed = value

    @property
    def speed(self) -> float:
        return round(math.sqrt(self._v_speed ** 2 + self._h_speed ** 2), 2)

    @property
    def rotate(self) -> int:
        return self._rotate

    @rotate.setter
    def rotate(self, value: int):
        self._rotate = value

    @property
    def power(self) -> int:
        return self._power

    @power.setter
    def power(self, value: int):
        self._power = value

    @property
    def fuel(self) -> int:
        return self._fuel

    @fuel.setter
    def fuel(self, value: int):
        self._fuel = value

    @property
    def crashed(self) -> bool:
        return self._crashed

    @crashed.setter
    def crashed(self, value: bool):
        self._crashed = value

    @property
    def crashed_on_landing_zone(self) -> bool:
        return self._crash_on_landing_zone

    @crashed_on_landing_zone.setter
    def crashed_on_landing_zone(self, value: bool):
        self._crash_on_landing_zone = value

    @property
    def crashed_distance(self) -> float:
        return self._crash_distance

    @crashed_distance.setter
    def crashed_distance(self, value: float):
        self._crash_distance = value

    @property
    def successfull_landing(self) -> bool:
        return self.crashed_on_landing_zone and abs(self.v_speed) < MAX_V_SPEED and abs(self.h_speed) < MAX_H_SPEED and self.rotate == 0 and self.fuel > 0

    def trajectory(self) -> list[tuple[Point, int, int]]:
        return self._trajectory

    def _compute(self) -> tuple:
        theta = math.radians(self._rotate + 90)
        acc_v = (math.sin(theta) * self._power) - MARS_GRAVITY
        acc_h = math.cos(theta) * self._power
        disp_v = self._v_speed + (acc_v / 2)
        disp_h = self._h_speed + (acc_h / 2)
        return acc_v, disp_v, acc_h, disp_h

    def _add_rotate(self, rotate: int):
        # Clamp the rotate in the range [-15, 15]
        rotate_step = clamp(rotate, -STEP_ROTATION_ANGLE, STEP_ROTATION_ANGLE)
        # Add the rotate angle and clamp in the range [-90, 90]
        self._rotate = clamp(self._rotate + rotate_step, MIN_ROTATION_ANGLE, MAX_ROTATION_ANGLE)

    def _add_power(self, power: int):
        # Clamp the power in the range [-1, 1]
        power_step = clamp(power, -STEP_POWER, STEP_POWER)
        # Add the power angle and clamp in the range [0, 4]
        self._power = clamp(self._power + power_step, MIN_POWER, MAX_POWER)

    def simulate(self, rotate: int, power: int):
        # Apply Rotate and Power
        self._add_rotate(rotate)
        self._add_power(power)

        # Decrease the fuel
        self._fuel -= self._power

        # Compute data
        acc_v, disp_v, acc_h, disp_h = self._compute()

        # Update position and speed
        self.position = self.position + Point(disp_h, disp_v)
        self._h_speed += acc_h
        self._v_speed += acc_v


class Gene:
    def __init__(self, rotate: int = 0, power: int = 0) -> None:
        self._rotate: int = rotate  # The change in angle, based on previous postion, in range [-15, 15]
        self._power: int = power    # The change in power, based on previoud postion, in range [-1, 1]

    def __eq__(self, other) -> bool:
        return self._rotate == other.rotate and self._power == other.power

    def __str__(self) -> str:
        return f"{self._rotate}, {self._power}"

    @property
    def rotate(self) -> int:
        return self._rotate

    @rotate.setter
    def rotate(self, value: int):
        self._rotate = value

    @property
    def power(self) -> int:
        return self._power

    @power.setter
    def power(self, value: int):
        self._power = value

    def random(self):
        self._rotate = randint(-STEP_ROTATION_ANGLE, STEP_ROTATION_ANGLE)
        self._power = randint(-STEP_POWER, STEP_POWER)


class Chromosome:
    def __init__(self, genes: Optional[list[Gene]] = None) -> None:
        self._genes: list[Gene] = [Gene() for _ in range(CHROMOSOME_SIZE)] if genes is None else genes
        self._surface: Optional[Surface] = None
        self._shuttle: Optional[Shuttle] = None
        self._evaluation: float = 0.0

    def __str__(self) -> str:
        msg: str = f"Evaluation: {round(self._evaluation, 2)}, Genes: "
        for idx, gene in enumerate(self._genes):
            msg += f"G{idx}: [{gene}]; "
        return msg[:-2]

    def __eq__(self, other) -> bool:
        return self._genes == other.genes

    @property
    def fitness(self) -> float:
        return self._evaluation

    @fitness.setter
    def fitness(self, value: float):
        self._evaluation = value

    def random(self):
        for gene in self._genes:
            gene.random()

    def set_gene(self, idx: int, gene: Gene):
        self._genes[idx] = gene

    def get_gene(self, idx: int) -> Gene:
        return self._genes[idx]

    @property
    def genes(self) -> list[Gene]:
        return self._genes

    @genes.setter
    def genes(self, genes: list[Gene]):
        self._genes = genes

    @property
    def surface(self) -> Surface:
        return self._surface

    @surface.setter
    def surface(self, surface: Surface):
        self._surface = surface

    @property
    def shuttle(self) -> Shuttle:
        return self._shuttle

    @shuttle.setter
    def shuttle(self, shuttle: Shuttle):
        self._shuttle = deepcopy(shuttle)

    def simulate(self):
        for gene in self._genes:
            self._shuttle.simulate(gene.rotate, gene.power)
            (
                crashed,
                idx_line_crashed,
                crashed_in_landing_zone,
            ) = self._surface.collision_with_surface(self._shuttle.trajectory()[-2][0], self._shuttle.trajectory()[-1][0])
            if crashed:
                self._shuttle.crashed = True
                distance = self._surface.find_distance_to_landing_zone(
                    self._shuttle.position, idx_line_crashed
                )
                self._shuttle.crashed_distance = distance
                if crashed_in_landing_zone:
                    self._shuttle.crashed_on_landing_zone = True
                    if self._shuttle.successfull_landing:
                        print("### Shuttle landing successfully ###")
                    # else:
                        # print("Shuttle crash on the landing area")
                break

    def evaluate(self):
        if (not (0 <= self._shuttle.position.x < MAP_WIDTH) or not (0 <= self._shuttle.position.y < MAP_HEIGHT)) or not self._shuttle.crashed:
            # 0: crashed out of the MAP
            self._evaluation = 0.0
        else:
            cost = 0.001
            cost += (self._shuttle.crashed_distance/500)**2
            # cost += (self._shuttle.v_speed/MAX_V_SPEED)**2
            # cost += (self._shuttle.h_speed/MAX_H_SPEED)**2
            # cost += (self._shuttle.rotate)**2
            # cost += (500 - self._shuttle.fuel)/500
            self._evaluation = 1/cost
            # print(f"evaluation: {self._evaluation:.3f}, cost: {cost:.3f}, distance: {self._shuttle.crashed_distance:.3f}, v_speed: {self._shuttle.v_speed:.3f}, h_speed: {self._shuttle.h_speed:.3f}, rotate: {self._shuttle.rotate:.3f}, fuel: {self._shuttle.fuel}")

    def mutate(self):
        for gene in self._genes:
            if random() < CHANCE_TO_MUTATE:
                gene.random()

    @staticmethod
    def from_crossover(mother, father) -> tuple:
        child0 = Chromosome()
        child1 = Chromosome()
        for idx in range(CHROMOSOME_SIZE):
            gene_mother = mother.get_gene(idx)
            gene_father = father.get_gene(idx)
            beta = random()
            child0_rotate = (beta * gene_mother.rotate) + ((1 - beta) * gene_father.rotate)
            child0_power = (beta * gene_mother.power) + ((1 - beta) * gene_father.power)
            child0.set_gene(idx, Gene(int(child0_rotate), int(child0_power)))
            child1_rotate = ((1 - beta) * gene_mother.rotate) + (beta * gene_father.rotate)
            child1_power = ((1 - beta) * gene_mother.power) + (beta * gene_father.power)
            child1.set_gene(idx, Gene(int(child1_rotate), int(child1_power)))
        return child0, child1


class GeneticPopulation:
    def __init__(self) -> None:
        self._population: list[Chromosome] = [Chromosome() for _ in range(POPULATION_SIZE)]
        self._surface: Optional[Surface] = None
        self._shuttle: Optional[Shuttle] = None

    def __str__(self) -> str:
        msg: str = ""
        for idx, chromosome in enumerate(self._population):
            msg += f"C{idx}: {chromosome}\n"
        return msg[:-2]

    def random(self):
        for chromosome in self._population:
            chromosome.random()

    def get_population(self) -> list[Chromosome]:
        return self._population

    @property
    def surface(self) -> Surface:
        return self._surface

    @surface.setter
    def surface(self, surface: Surface):
        self._surface = surface

    @property
    def shuttle(self) -> Shuttle:
        return self._shuttle

    @shuttle.setter
    def shuttle(self, shuttle: Shuttle):
        self._shuttle = shuttle

    def simulate(self):
        for chromosome in self._population:
            chromosome.surface = self._surface
            chromosome.shuttle = self._shuttle
            chromosome.simulate()
            chromosome.evaluate()

        # Sort by attribute evaluation
        self._population.sort(key=lambda x: x.fitness, reverse=True)

    def make_next_generation(self):

        # Filter the top graded chromosome
        cut_pos = int(GRADED_RETAIN_PERCENT * POPULATION_SIZE)
        new_population = deepcopy(self._population[:cut_pos])

        # Randomly add other chromosome to promote genetic diversity
        for chromosome in self._population[cut_pos:]:
            if random() < CHANCE_RETAIN_NONGRATED:
                new_population.append(chromosome)

        # Crossover parents to create children
        children = []
        while (len(new_population) + len(children)) < POPULATION_SIZE:
            mother = choice(new_population)
            father = choice(new_population)
            if mother == father:
                continue
            child1, child2 = Chromosome.from_crossover(mother, father)
            children.append(deepcopy(child1))
            children.append(deepcopy(child2))

        # Mutate some chromosome
        for chromosome in children:
            chromosome.mutate()

        new_population.extend(children)

        # Save the new population
        self._population = deepcopy(new_population[:POPULATION_SIZE])


class Renderer:
    def __init__(self) -> None:
        sdl2.ext.init()
        self._window = sdl2.ext.Window("2D drawing primitives", size=(MAP_WIDTH//DIVISOR_FOR_PRINTER, MAP_HEIGHT//DIVISOR_FOR_PRINTER))
        self._window.show()
        self._color = sdl2.ext.Color()

    def refresh(self):
        self._window.refresh()

    def clear(self):
        sdl2.ext.fill(self._window.get_surface(), sdl2.ext.Color(0, 0, 0, 0))

    def set_color(self, color: sdl2.ext.Color):
        self._color = color

    def draw_line(self, line: Line):
        sdl2.ext.line(self._window.get_surface(), self._color, line.line_to_point())

    def draw_lines(self, points: list[Point]):
        for idx, point in enumerate(points):
            if idx != 0:
                p1 = point
                line = Line(p0, p1)
                self.draw_line(line)
            p0 = point

    def draw_shuttle_trajectory(self, shuttle: Shuttle):
        self.draw_lines([x[0] for x in shuttle.trajectory()])

    def is_quit(self):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                return True

        return False


class Game:
    def __init__(self) -> None:
        self._surface: Surface = Surface()
        self._shuttle: Shuttle = Shuttle()
        self._genetic_population: GeneticPopulation = GeneticPopulation()
        self._renderer: Renderer = Renderer()
        self._log_fitness: list[float] = []
        self._chromosome_landing: list[Chromosome] = []

    def _read_parameters_from_file(self, filename):
        data = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                data.append(line.rsplit())

        # Create surface from data
        self._surface.create_surface_from_point(data[:-1])
        self._genetic_population.surface = self._surface

        # Create shuttle from data
        self._shuttle.position = Point(int(data[-1][0]), int(data[-1][1]))
        self._shuttle.h_speed = int(data[-1][2])
        self._shuttle.v_speed = int(data[-1][3])
        self._shuttle.fuel = int(data[-1][4])
        self._shuttle.rotate = int(data[-1][5])
        self._shuttle.power = int(data[-1][6])
        self._genetic_population.shuttle = self._shuttle

    def _write_svg(self, filename:str):
        line_chart = pygal.Line(show_dots=True, show_legend=False)
        line_chart.title = 'Fitness evolution'
        line_chart.x_title = 'Generations'
        line_chart.y_title = 'Fitness'
        line_chart.add('Fitness', self._log_fitness)
        line_chart.render_to_file(filename)

    def _write_data_to_file(self, filename: str):
        if len(self._chromosome_landing) > 0:
            result = "{\n"
            result += f"\"{filename.split('.')[0]}\" : [\n"
            for gene in self._chromosome_landing[0].genes:
                result += f"[{gene}],\n"
            result = result[:-2]
            result += "\n"
            result += "]\n"
            result += "}"
            with open(filename, "w") as f:
                f.write(result)

    def _begin(self, filename: str):
        self._read_parameters_from_file(filename)
        print("Start shuttle => " + str(self._shuttle))
        self._genetic_population.random()

    def _loop(self):
        done: bool = False
        nb_generation = 0
        while not done and not self._renderer.is_quit() and nb_generation < GENERATION_COUNT_MAX:
            # Simulate the genetic population
            self._genetic_population.simulate()

            self._log_fitness.append(self._genetic_population.global_fitness())

            # Create the renderer
            self._renderer.clear()
            for line in self._surface.lines:
                self._renderer.set_color(sdl2.ext.Color())
                self._renderer.draw_line(line)
            for chromosome in self._genetic_population.get_chromosomes():
                self._renderer.draw_shuttle_trajectory(chromosome.shuttle)
            self._renderer.refresh()

            # Check if one or more shuttle are successfully landing
            for chromosome in self._genetic_population.get_population():
                if chromosome.shuttle.successfull_landing:
                    print("The shuttle landing with:")
                    print(chromosome.shuttle)
                    done = True
                    result = ""
                    for gene in chromosome.genes:
                        result += f"{gene}\n"
                    with open(f"result.txt", "w") as f:
                        f.write(result)
                    continue

            nb_generation += 1
            print(f"nb_generation: {nb_generation}")
            # with open(f"population_{nb_generation}.txt", "w") as f:
                # f.write(str(self._genetic_population))

            # Create a new genetic population
            self._genetic_population.make_next_generation()
            time.sleep(0.5)
            # done = True
            # while not self._renderer.is_quit():
            #     pass

        if not done:
            print("!!! No solution found !!!")

    def _end(self, filename: str):
        filename = filename.replace("input", "output")
        self._write_data_to_file(filename)
        self._write_svg(filename.replace("txt", "svg"))
        while not self._renderer.is_quit():
            pass

    def play(self, filename):
        self._begin(filename)
        self._loop()
        self._end(filename)


class Simulator:
    def __init__(self) -> None:
        self._surface: Surface = Surface()
        self._shuttle: Shuttle = Shuttle()
        self._chromosome: Chromosome = Chromosome()
        self._renderer: Renderer = Renderer()

    def _read_parameters_from_file(self, filename):
        data = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                data.append(line.rsplit())

        # Create surface from data
        self._surface.create_surface_from_point(data[:-1])

        # Create shuttle from data
        self._shuttle.position = Point(int(data[-1][0]), int(data[-1][1]))
        self._shuttle.h_speed = int(data[-1][2])
        self._shuttle.v_speed = int(data[-1][3])
        self._shuttle.fuel = int(data[-1][4])
        self._shuttle.rotate = int(data[-1][5])
        self._shuttle.power = int(data[-1][6])

    def add_data(self, filename: str):
        with open(filename, "r") as f:
            data = json.load(f)
        data = data.get(filename.removesuffix(".txt"))
        for idx, d in enumerate(data):
            gene = Gene(int(d[0]), int(d[1]))
            self._chromosome.set_gene(idx, gene)

    def _begin(self, filename: str):
        self._read_parameters_from_file(filename)
        print("Start shuttle => " + str(self._shuttle))

        self._chromosome.shuttle = self._shuttle
        self._chromosome.surface = self._surface

        # Create the renderer
        self._renderer.clear()
        for line in self._surface.lines:
            if line.landing_zone_direction == 0:
                self._renderer.set_color(sdl2.ext.Color(255, 0, 255))
                self._renderer.draw_line(line)
            else:
                self._renderer.set_color(sdl2.ext.Color(255, 0, 0))
                self._renderer.draw_line(line)

    def _loop(self):
        self._chromosome.simulate(True)
        self._chromosome.evaluate()

    def _end(self):
        # Create the renderer
        self._renderer.draw_shuttle_trajectory(self._chromosome.shuttle)
        self._renderer.refresh()
        while not self._renderer.is_quit():
            pass

    def play(self, filename: str):
        self._begin(filename)
        self._loop()
        self._end()


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) >= 2 else "input_2_1.txt"
    file_replay = sys.argv[2] if len(sys.argv) >= 3 else None

    if file_replay is not None:
        simu = Simulator()
        simu.add_data(file_replay)
        simu.play(filename)

    else:
        game = Game()
        game.play(filename)

    exit()
