"""Вещественный генетический алгоритм и случайный поиск для варианта 14."""

from dataclasses import dataclass
from random import Random


DIMENSION = 7
LOWER = -5.0
UPPER = 5.0


@dataclass(frozen=True)
class Config:
    population_size: int = 60
    generations: int = 150
    elite_count: int = 2
    tournament_size: int = 3
    crossover_probability: float = 0.9
    blend_alpha: float = 0.5
    mutation_probability: float = 0.10
    mutation_sigma: float = 0.2

    @property
    def evaluation_budget(self) -> int:
        return self.population_size + self.generations * (
            self.population_size - self.elite_count
        )


@dataclass(frozen=True)
class Individual:
    coordinates: tuple[float, ...]
    value: float


@dataclass(frozen=True)
class Run:
    best: Individual
    history: tuple[float, ...]
    evaluations: int


def objective(coordinates: tuple[float, ...]) -> float:
    return 0.5 * sum(x**4 - 16 * x**2 + 5 * x for x in coordinates)


def sample(rng: Random) -> Individual:
    coordinates = tuple(rng.uniform(LOWER, UPPER) for _ in range(DIMENSION))
    return Individual(coordinates, objective(coordinates))


def select(population: list[Individual], size: int, rng: Random) -> Individual:
    return min(rng.sample(population, size), key=lambda individual: individual.value)


def crossover(
    first: tuple[float, ...], second: tuple[float, ...], alpha: float, rng: Random
) -> tuple[float, ...]:
    child = []
    for left, right in zip(first, second):
        span = abs(left - right)
        child.append(
            rng.uniform(min(left, right) - alpha * span, max(left, right) + alpha * span)
        )
    return tuple(child)


def mutate(coordinates: tuple[float, ...], config: Config, rng: Random) -> tuple[float, ...]:
    child = []
    for x in coordinates:
        if rng.random() < config.mutation_probability:
            x += rng.gauss(0, config.mutation_sigma)
        child.append(min(UPPER, max(LOWER, x)))
    return tuple(child)


def genetic_algorithm(config: Config, seed: int) -> Run:
    rng = Random(seed)
    population = [sample(rng) for _ in range(config.population_size)]
    evaluations = len(population)
    best = min(population, key=lambda individual: individual.value)
    history = [best.value]

    for _ in range(config.generations):
        ranked = sorted(population, key=lambda individual: individual.value)
        next_population = ranked[: config.elite_count]
        while len(next_population) < config.population_size:
            first = select(population, config.tournament_size, rng).coordinates
            coordinates = first
            if rng.random() < config.crossover_probability:
                second = select(population, config.tournament_size, rng).coordinates
                coordinates = crossover(first, second, config.blend_alpha, rng)
            coordinates = mutate(coordinates, config, rng)
            next_population.append(Individual(coordinates, objective(coordinates)))
            evaluations += 1

        population = next_population
        best = min(population, key=lambda individual: individual.value)
        history.append(best.value)

    return Run(best, tuple(history), evaluations)


def random_search(config: Config, seed: int) -> Run:
    rng = Random(seed)
    best = min(
        (sample(rng) for _ in range(config.population_size)),
        key=lambda individual: individual.value,
    )
    evaluations = config.population_size
    history = [best.value]
    batch_size = config.population_size - config.elite_count

    for _ in range(config.generations):
        for _ in range(batch_size):
            candidate = sample(rng)
            if candidate.value < best.value:
                best = candidate
        evaluations += batch_size
        history.append(best.value)

    return Run(best, tuple(history), evaluations)
