import random
import pandas as pd
from src.simulator import Simulator

class GeneticAlgorithm:
    def __init__(self, ceps_df: pd.DataFrame, wind_schedule, pop_size=10, generations=20, mutation_rate=0.2):
        self.ceps_df = ceps_df
        self.wind_schedule = wind_schedule
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initial_population(self, n_points):
        ceps = list(self.ceps_df["cep"])
        population = []
        for _ in range(self.pop_size):
            route = random.sample(ceps, n_points)
            population.append(route)
        return population

    def fitness(self, route):
        simulator = Simulator(self.ceps_df, self.wind_schedule)
        result = simulator.simulate_route(route)
        # Menor tempo + menor custo = melhor (inverso)
        score = 1 / (result["tempo_total_h"] + 0.5 * result["custo_total"])
        return score, result

    def selection(self, population):
        """Seleciona os 2 melhores indivíduos"""
        scores = [(self.fitness(ind)[0], ind) for ind in population]
        scores.sort(reverse=True, key=lambda x: x[0])
        return [scores[0][1], scores[1][1]]

    def crossover(self, parent1, parent2):
        """Crossover simples: fatia e combina"""
        size = len(parent1)
        a, b = sorted(random.sample(range(size), 2))
        child = parent1[a:b] + [x for x in parent2 if x not in parent1[a:b]]
        return child

    def mutate(self, route):
        """Troca 2 posições aleatórias"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def run(self, n_points=8):
        population = self.initial_population(n_points)
        best_route = None
        best_score = 0
        best_result = None

        for _ in range(self.generations):
            new_population = []
            for _ in range(self.pop_size):
                parents = self.selection(population)
                child = self.crossover(parents[0], parents[1])
                child = self.mutate(child)
                new_population.append(child)

            population = new_population

            # Avalia o melhor
            for route in population:
                score, result = self.fitness(route)
                if score > best_score:
                    best_score = score
                    best_route = route
                    best_result = result

        return best_route, best_result
