from dataclasses import dataclass
import math
from random import randint, random, choice
from typing import Optional
from copy import deepcopy
import sdl2
import sdl2.ext
import time

from sdl2.ext import color

INPUT_DATA = {
    "surface": [
        [0, 100],
        [1000, 500],
        [1500, 100],
        [3000, 100],
        [5000, 1500],
        [6999, 1000],
    ],
    "shuttle": {
        "position": [2500, 2500],
        "h_speed": 0,
        "v_speed": 0,
        "fuel": 500,
        "rotate": 0,
        "power": 0
    },
}

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
    x: int
    y: int

    def __str__(self) -> str:
        return f"({self.x},{self.y})"

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
        # self._is_landing_zone: bool = False

    @property
    def point0(self) -> int:
        return self._point0

    @property
    def point1(self) -> int:
        return self._point1

    def line_to_point(self) -> tuple:
        return (self._point0.x//DIVISOR_FOR_PRINTER,
                MAP_HEIGHT//DIVISOR_FOR_PRINTER - self._point0.y//DIVISOR_FOR_PRINTER,
                self._point1.x//DIVISOR_FOR_PRINTER,
                MAP_HEIGHT//DIVISOR_FOR_PRINTER - self._point1.y//DIVISOR_FOR_PRINTER)

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
            point1 = Point(point[0], point[1])
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
            print("Shuttle is out of the MAP")
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
        self._crash_on_landing_zone: bool = False
        self._crash_distance: float = 0.0  # Distance between crash and landing zone
        self._successfull_landing: bool = False
        self._trajectory: list[Point] = []

    def __str__(self) -> str:
        return (
            f"pos: {self.position}, "
            + f"h_speed: {round(self._h_speed, 2)}, "
            + f"v_speed: {round(self._v_speed, 2)}, "
            + f"rotate: {self._rotate}, "
            + f"power: {self._power}, "
            + f"fuel: {self._fuel}"
        )

    @property
    def position(self) -> Point:
        return self._trajectory[-1]

    @position.setter
    def position(self, value: Point):
        self._trajectory.append(value)

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
        return self._successfull_landing

    @successfull_landing.setter
    def successfull_landing(self, value: bool):
        self._successfull_landing = value

    def trajectory(self) -> list[Point]:
        return self._trajectory

    def compute(self) -> tuple:
        theta = math.radians(90 + self._rotate)
        acc_v = (math.sin(theta) * self._power) - MARS_GRAVITY
        acc_h = math.cos(theta) * self._power
        disp_v = int(self._v_speed + (acc_v * 0.5))
        disp_h = int(self._h_speed + (acc_h * 0.5))
        return acc_v, disp_v, acc_h, disp_h

    def _add_rotate(self, rotate: int):
        # Clamp the rotate in the range [-15, 15]
        rotate_step = clamp(rotate, -STEP_ROTATION_ANGLE, STEP_ROTATION_ANGLE)
        # Add the rotate angle and clamp in the range [-90, 90]
        self._rotate = clamp(
            self._rotate + rotate_step, MIN_ROTATION_ANGLE, MAX_ROTATION_ANGLE
        )

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
        acc_v, disp_v, acc_h, disp_h = self.compute()

        # Update position and speed
        self.position = self.position + Point(disp_h, disp_v)
        self._h_speed += acc_h
        self._v_speed += acc_v

    @staticmethod
    def is_good_landing(sht0) -> bool:
        if (
            abs(sht0.v_speed) < MAX_V_SPEED
            and abs(sht0.h_speed) < MAX_H_SPEED
            and abs(sht0.rotate) < STEP_ROTATION_ANGLE
        ):
            # if abs(sht1.v_speed) < MAX_V_SPEED and abs(sht1.h_speed) < MAX_H_SPEED:
            return True

        return False


class Gene:
    def __init__(self, rotate: int = 0, power: int = 0) -> None:
        self._rotate: int = rotate
        self._power: int = power

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

    def __eq__(self, other) -> bool:
        return self._rotate == other.rotate and self._power == other.power

    def __str__(self) -> str:
        return f"{self._rotate}, {self._power}"

    def clamp(self):
        self._rotate = clamp(self._rotate, MIN_ROTATION_ANGLE, MAX_ROTATION_ANGLE)
        self._power = clamp(self._power, MIN_POWER, MAX_POWER)

    def random(self):
        self._rotate = randint(MIN_ROTATION_ANGLE, MAX_ROTATION_ANGLE)
        self._power = randint(MIN_POWER, MAX_POWER)


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
    def evaluation(self) -> float:
        return self._evaluation

    @evaluation.setter
    def evaluation(self, value: float):
        self._evaluation = value

    def random(self):
        for gene in self._genes:
            gene.random()

    def set_gene(self, idx: int, gene: Gene):
        self._genes[idx] = gene

    def get_gene(self, idx: int) -> Gene:
        return self._genes[idx]

    @property
    def genes(self) -> Gene:
        return self._genes

    @genes.setter
    def genes(self, genes: Gene):
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
        for idx, gene in enumerate(self._genes):
            # print(f"shuttle: {self._shuttle}")
            self._shuttle.simulate(gene.rotate, gene.power)
            (
                crashed,
                idx_line_crashed,
                crashed_in_landing_zone,
            ) = self._surface.collision_with_surface(self._shuttle.trajectory()[-2], self._shuttle.trajectory()[-1])
            if crashed:
                # print(f"crashed at the gene {idx}")
                if crashed_in_landing_zone:
                    self._shuttle.crashed_on_landing_zone = True
                    if Shuttle.is_good_landing(self._shuttle):
                        self._shuttle.successfull_landing = True
                        print("Shuttle landing successfully")
                    else:
                        print("Shuttle crash on the landing area")
                else:
                    distance = self._surface.find_distance_to_landing_zone(
                        self._shuttle.position, idx_line_crashed
                    )
                    self._shuttle.crashed_distance = distance
                break

    def evaluate(self):
        if (not 0 <= self._shuttle.position.x < MAP_WIDTH or not 0 <= self._shuttle.position.y < MAP_HEIGHT):
            # 0: crashed out of the MAP
            self._evaluation = 0.0
        else:
            eval_distance = clamp(3000 - (abs(self._shuttle.crashed_distance) - 500), 0, 3000) / 3000
            eval_v_speed = clamp(MAX_V_SPEED - abs(self._shuttle.v_speed), 0, MAX_V_SPEED) / MAX_V_SPEED
            eval_h_speed = clamp(MAX_H_SPEED - abs(self._shuttle.h_speed), 0, MAX_H_SPEED) / MAX_H_SPEED
            eval_rotate = clamp(MAX_ROTATION_ANGLE - abs(self._shuttle.rotate), 0, MAX_ROTATION_ANGLE) / MAX_ROTATION_ANGLE
            self._evaluation = (eval_distance + eval_v_speed + eval_h_speed + eval_rotate) / 4

            print(f"evaluation: {self._evaluation:.3f}, distance: {eval_distance:.3f}, v_speed: {eval_v_speed:.3f}, h_speed: {eval_h_speed:.3f}, rotate: {eval_rotate:.3f}")

        # # 0-100: crashed somewhere, calculate score by distance to landing area
        # elif not self._shuttle.crashed_on_landing_zone:
        #     distance = self._shuttle.crashed_distance

        #     # Calculate score from distance
        #     max_distance = self._surface.get_max_distance()
        #     self._evaluation = 100.0 - (100.0 * distance / max_distance)

        #     # High speeds are bad, they decrease maneuvrability
        #     # self._evaluation -= clamp(self._shuttle.speed * 0.1, 0, 100)

        # # 100-200: crashed into landing area, calculate score by speed above safety
        # elif not self._shuttle.successfull_landing:
        #     self._evaluation = 200.0 - clamp(self._shuttle.speed * 0.5, 0, 100)

        # # 200-300: landed safely, calculate score by fuel remaining
        # else:
        #     self._evaluation = 200.0 + self._shuttle.fuel

    def mutate(self):
        for gene in self._genes:
            if random() < CHANCE_TO_MUTATE:
                gene.random()

    @staticmethod
    def from_crossover(mother, father):
        cut = randint(1, CHROMOSOME_SIZE-1)
        child0 = Chromosome(mother.genes[:cut] + father.genes[cut:])
        child1 = Chromosome(father.genes[:cut] + mother.genes[cut:])
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

    def get_chromosomes(self) -> list[Chromosome]:
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

    def make_next_generation(self):
        # Sort by attribute evaluation
        self._population.sort(key=lambda x: x.evaluation, reverse=True)
        for chromosome in self._population:
            print(f"evaluation: {chromosome.evaluation:.3f}")

        # Filter the top graded chromosome
        cut_pos = int(GRADED_RETAIN_PERCENT * POPULATION_SIZE)
        new_population = self._population[:cut_pos]

        print("len pop after cut: " + str(len(new_population)))

        # Randomly add other chromosome to promote genetic diversity
        for chromosome in self._population[cut_pos:]:
            if random() < CHANCE_RETAIN_NONGRATED:
                new_population.append(chromosome)

        print("len pop before crossover: " + str(len(new_population)))

        # Crossover parents to create children
        while len(new_population) < POPULATION_SIZE:
            mother = choice(new_population)
            father = choice(new_population)
            if mother == father:
                continue
            child1, child2 = Chromosome.from_crossover(mother, father)
            new_population.append(child1)
            new_population.append(child2)

        print("len pop after crossover: " + str(len(new_population)))

        # Mutate some chromosome
        for chromosome in new_population:
            chromosome.mutate()

        # Save the new population
        self._population = new_population[:POPULATION_SIZE]


class Renderer:
    def __init__(self) -> None:
        sdl2.ext.init()
        self._window = sdl2.ext.Window("2D drawing primitives", size=(MAP_WIDTH//DIVISOR_FOR_PRINTER, MAP_HEIGHT//DIVISOR_FOR_PRINTER))
        self._window.show()

    def refresh(self):
        self._window.refresh()

    def clear(self):
        sdl2.ext.fill(self._window.get_surface(), sdl2.ext.Color(0, 0, 0, 0))

    def draw_line(self, line: Line):
        color = sdl2.ext.Color()
        sdl2.ext.line(self._window.get_surface(), color, line.line_to_point())

    def draw_lines(self, points: list[Point]):
        for idx, point in enumerate(points):
            if idx != 0:
                p1 = point
                line = Line(p0, p1)
                self.draw_line(line)
            p0 = point

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

    def add_parameters(self, data):
        # Create surface from data
        surface = data.get("surface", None)
        if surface is not None:
            self._surface.create_surface_from_point(surface)
            self._genetic_population.surface = self._surface

        # Create shuttle from data
        shuttle = data.get("shuttle", None)
        if shuttle is not None:
            self._shuttle.position = Point(int(shuttle["position"][0]), int(shuttle["position"][1]))
            self._shuttle.h_speed = int(shuttle["h_speed"])
            self._shuttle.v_speed = int(shuttle["v_speed"])
            self._shuttle.fuel = int(shuttle["fuel"])
            self._shuttle.rotate = int(shuttle["rotate"])
            self._genetic_population.shuttle = self._shuttle

    def begin(self):
        print("Start shuttle => " + str(self._shuttle))
        self._genetic_population.random()
        for line in self._surface.lines:
            self._renderer.draw_line(line)

    def loop(self):
        done: bool = False
        nb_generation = 0
        while not done and not self._renderer.is_quit() and nb_generation < GENERATION_COUNT_MAX:
            # Simulate the genetic population
            self._genetic_population.simulate()

            # Create the renderer
            self._renderer.clear()
            for line in self._surface.lines:
                self._renderer.draw_line(line)
            for chromosome in self._genetic_population.get_chromosomes():
                self._renderer.draw_lines(chromosome.shuttle.trajectory())
            self._renderer.refresh()

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

    def end(self):
        pass

    def play(self):
        self.begin()
        self.loop()
        self.end()


if __name__ == "__main__":
    game = Game()
    game.add_parameters(INPUT_DATA)

    game.play()
    time.sleep(2)