"""Core optimization routines used for spatial Wind-Solar-BESS siting.

The decision vector is binary: 1 means that a candidate grid cell is selected
for development and 0 means that it is not selected. Objectives are returned in
minimization form: negative energy production, negative net return, and grid
connection distance.

The offspring operator is DE-inspired and adapted to binary vectors. It should
not be interpreted as a continuous-space differential evolution operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class ObjectiveData:
    """Arrays for one technology/pathway scenario over valid candidate cells."""

    power_kwh_per_hour: np.ndarray
    lcoe_cny_per_kwh: np.ndarray
    local_demand_kwh: np.ndarray
    grid_distance: np.ndarray
    sale_price_cny_per_kwh: float = 0.341
    transmission_penalty_cny_per_kwh: float = 0.1

    @property
    def dimension(self) -> int:
        return int(self.local_demand_kwh.shape[0])


@dataclass
class AlgorithmResult:
    population: np.ndarray
    fitness: np.ndarray
    history: List[np.ndarray]
    generations: List[int]
    d_indicator: List[float]


def objective_values(solution: np.ndarray, data: ObjectiveData) -> np.ndarray:
    """Return three minimization objectives for one binary siting solution."""

    annual_generation = solution * data.power_kwh_per_hour * 8760.0
    surplus = np.maximum(data.power_kwh_per_hour * 8760.0 - data.local_demand_kwh, 0.0)
    served_locally = np.minimum(data.power_kwh_per_hour * 8760.0, data.local_demand_kwh)

    net_return = np.where(
        data.power_kwh_per_hour * 8760.0 > data.local_demand_kwh,
        solution
        * (
            (data.sale_price_cny_per_kwh - data.transmission_penalty_cny_per_kwh - data.lcoe_cny_per_kwh)
            * surplus
            + (data.sale_price_cny_per_kwh - data.lcoe_cny_per_kwh) * served_locally
        ),
        solution * (data.sale_price_cny_per_kwh - data.lcoe_cny_per_kwh) * annual_generation,
    )

    energy = float(np.sum(annual_generation))
    return_cny = float(np.sum(net_return))
    distance = float(np.sum(solution * data.grid_distance))
    return np.array([-energy, -return_cny, distance], dtype=float)


def initialize_population(pop_size: int, dimension: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=(pop_size, dimension), dtype=np.int8)


def spatial_discrimination(population: np.ndarray) -> float:
    """D indicator used to summarize departure from random binary allocation."""

    pop_size = population.shape[0]
    return float(np.mean(np.abs(np.sum(population, axis=0) - pop_size / 2.0) / (pop_size / 2.0)) * 100.0)


def binary_de_offspring(
    population: np.ndarray,
    index: int,
    rng: np.random.Generator,
    crossover_rate: float,
    mutation_factor: float,
    candidate_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Generate one binary offspring using a DE-inspired recombination step."""

    if candidate_indices is None:
        candidate_indices = [i for i in range(len(population)) if i != index]
    a, b, c = population[rng.choice(candidate_indices, 3, replace=False)]
    mutant = np.clip(a + mutation_factor * (b - c), 0, 1)
    crossover_mask = rng.random(len(mutant)) < crossover_rate
    offspring = np.where(crossover_mask, mutant, population[index])
    return np.rint(offspring).astype(np.int8)


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(fitness: np.ndarray) -> List[List[int]]:
    dominated_sets = [[] for _ in range(len(fitness))]
    domination_counts = np.zeros(len(fitness), dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(len(fitness)):
        for q in range(len(fitness)):
            if dominates(fitness[p], fitness[q]):
                dominated_sets[p].append(q)
            elif dominates(fitness[q], fitness[p]):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def crowding_distance(fitness: np.ndarray, front: Sequence[int]) -> np.ndarray:
    distances = np.zeros(len(front), dtype=float)
    if not front:
        return distances

    front_fitness = np.array([fitness[i] for i in front])
    for objective in range(front_fitness.shape[1]):
        sorted_idx = np.argsort(front_fitness[:, objective])
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        f_min = front_fitness[sorted_idx[0], objective]
        f_max = front_fitness[sorted_idx[-1], objective]
        if f_max == f_min:
            continue
        for j in range(1, len(front) - 1):
            distances[sorted_idx[j]] += (
                front_fitness[sorted_idx[j + 1], objective] - front_fitness[sorted_idx[j - 1], objective]
            ) / (f_max - f_min)
    return distances


def _record_generation(
    population: np.ndarray,
    history: List[np.ndarray],
    generations: List[int],
    d_indicator: List[float],
    generation: int,
    record_generations: set[int],
) -> None:
    generation_number = generation + 1
    if generation_number in record_generations:
        history.append(population.copy())
    generations.append(generation_number)
    d_indicator.append(spatial_discrimination(population))


def nsga2(
    data: ObjectiveData,
    pop_size: int = 200,
    max_generations: int = 500,
    crossover_rate: float = 0.4,
    mutation_factor: float = 0.4,
    record_generations: Sequence[int] = (1, 50, 100, 200, 500),
    seed: int | None = None,
) -> AlgorithmResult:
    rng = np.random.default_rng(seed)
    population = initialize_population(pop_size, data.dimension, rng)
    fitness = np.array([objective_values(ind, data) for ind in population])
    history: List[np.ndarray] = []
    generations: List[int] = []
    d_values: List[float] = []
    record_set = set(record_generations)

    for generation in range(max_generations):
        offspring = np.array(
            [
                binary_de_offspring(population, i, rng, crossover_rate, mutation_factor)
                for i in range(pop_size)
            ],
            dtype=np.int8,
        )
        offspring_fitness = np.array([objective_values(ind, data) for ind in offspring])
        combined_population = np.vstack((population, offspring))
        combined_fitness = np.vstack((fitness, offspring_fitness))
        fronts = fast_non_dominated_sort(combined_fitness)

        new_population = []
        new_fitness = []
        for front in fronts:
            if len(new_population) + len(front) > pop_size:
                distance = crowding_distance(combined_fitness, front)
                for local_index in np.argsort(-distance):
                    if len(new_population) == pop_size:
                        break
                    original_index = front[local_index]
                    new_population.append(combined_population[original_index])
                    new_fitness.append(combined_fitness[original_index])
                break
            for original_index in front:
                new_population.append(combined_population[original_index])
                new_fitness.append(combined_fitness[original_index])

        population = np.array(new_population, dtype=np.int8)
        fitness = np.array(new_fitness, dtype=float)
        _record_generation(population, history, generations, d_values, generation, record_set)

    return AlgorithmResult(population, fitness, history, generations, d_values)


def _weight_vectors(pop_size: int, num_objectives: int, t_neighbors: int, rng: np.random.Generator):
    vectors = rng.random((pop_size, num_objectives))
    vectors /= np.sum(vectors, axis=1, keepdims=True)
    distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=2)
    neighborhoods = np.argsort(distances, axis=1)[:, :t_neighbors]
    return vectors, neighborhoods


def _tchebycheff(fitness: np.ndarray, ideal_point: np.ndarray, weight: np.ndarray) -> float:
    return float(np.max(weight * np.abs(fitness - ideal_point)))


def moead(
    data: ObjectiveData,
    pop_size: int = 200,
    max_generations: int = 500,
    crossover_rate: float = 0.4,
    mutation_factor: float = 0.4,
    t_neighbors: int = 20,
    record_generations: Sequence[int] = (1, 50, 100, 200, 500),
    seed: int | None = None,
) -> AlgorithmResult:
    rng = np.random.default_rng(seed)
    population = initialize_population(pop_size, data.dimension, rng)
    weights, neighborhoods = _weight_vectors(pop_size, 3, t_neighbors, rng)
    fitness = np.array([objective_values(ind, data) for ind in population])
    ideal_point = np.min(fitness, axis=0)
    history: List[np.ndarray] = []
    generations: List[int] = []
    d_values: List[float] = []
    record_set = set(record_generations)

    for generation in range(max_generations):
        for i in range(pop_size):
            offspring = binary_de_offspring(
                population,
                i,
                rng,
                crossover_rate,
                mutation_factor,
                candidate_indices=neighborhoods[i],
            )
            offspring_fitness = objective_values(offspring, data)
            ideal_point = np.minimum(ideal_point, offspring_fitness)
            for j in neighborhoods[i]:
                old_value = _tchebycheff(fitness[j], ideal_point, weights[j])
                new_value = _tchebycheff(offspring_fitness, ideal_point, weights[j])
                if new_value < old_value:
                    population[j] = offspring
                    fitness[j] = offspring_fitness
        _record_generation(population, history, generations, d_values, generation, record_set)

    return AlgorithmResult(population, fitness, history, generations, d_values)


def _spea2_fitness(fitness: np.ndarray) -> np.ndarray:
    n = len(fitness)
    strengths = np.zeros(n, dtype=float)
    raw = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            if dominates(fitness[i], fitness[j]):
                strengths[i] += 1.0
    for i in range(n):
        for j in range(n):
            if dominates(fitness[j], fitness[i]):
                raw[i] += strengths[j]

    distances = np.linalg.norm(fitness[:, None, :] - fitness[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    k = max(1, int(np.sqrt(n)))
    kth = np.sort(distances, axis=1)[:, k - 1]
    density = 1.0 / (kth + 2.0)
    return raw + density


def _environmental_selection(population: np.ndarray, fitness: np.ndarray, archive_size: int):
    score = _spea2_fitness(fitness)
    selected = np.argsort(score)[:archive_size]
    return population[selected], fitness[selected]


def spea2_with_de(
    data: ObjectiveData,
    pop_size: int = 200,
    archive_size: int | None = None,
    max_generations: int = 500,
    crossover_rate: float = 0.4,
    mutation_factor: float = 0.4,
    record_generations: Sequence[int] = (1, 50, 100, 200, 500),
    seed: int | None = None,
) -> AlgorithmResult:
    if archive_size is None:
        archive_size = pop_size

    rng = np.random.default_rng(seed)
    population = initialize_population(pop_size, data.dimension, rng)
    archive = np.empty((0, data.dimension), dtype=np.int8)
    archive_fitness = np.empty((0, 3), dtype=float)
    history: List[np.ndarray] = []
    generations: List[int] = []
    d_values: List[float] = []
    record_set = set(record_generations)

    for generation in range(max_generations):
        population_fitness = np.array([objective_values(ind, data) for ind in population])
        if len(archive):
            combined_population = np.vstack((population, archive))
            combined_fitness = np.vstack((population_fitness, archive_fitness))
        else:
            combined_population = population.copy()
            combined_fitness = population_fitness.copy()

        archive, archive_fitness = _environmental_selection(combined_population, combined_fitness, archive_size)
        parents = archive[rng.choice(len(archive), pop_size, replace=True)]
        neighborhoods = [np.delete(np.arange(len(parents)), i) for i in range(len(parents))]
        population = np.array(
            [
                binary_de_offspring(
                    parents,
                    i,
                    rng,
                    crossover_rate,
                    mutation_factor,
                    candidate_indices=neighborhoods[i],
                )
                for i in range(len(parents))
            ],
            dtype=np.int8,
        )

        _record_generation(archive, history, generations, d_values, generation, record_set)

    return AlgorithmResult(archive, archive_fitness, history, generations, d_values)
