from abc import ABC, abstractmethod

import numpy as np

class BaseEA(ABC):
	def __init__(self, population_size, minimize, mutation_rate, num_offspring):
		self.population_size = population_size
		self.minimize = minimize
		self.chromosomes = None
		self.curr_fitness = None
		self.mutation_rate = mutation_rate
		self.num_offspring = num_offspring

	# Problem-specific
	@abstractmethod
	def initialize_population(self):
		pass

	@abstractmethod
	def create_offspring(self):
		pass

	@abstractmethod
	def calculate_fitness(self, chromosome):
		pass

	@abstractmethod
	def select_to_kill(self):
		"""Return the integer index of the individual selected for replacement."""
		pass

	def evaluate_population(self):
		if self.chromosomes is None:
			raise ValueError("Population must be initialized before evaluation.")
		self.curr_fitness = np.array(
			[self.calculate_fitness(chromosome) for chromosome in self.chromosomes]
		)

	def best_index(self):
		if self.curr_fitness is None:
			raise ValueError("Population fitness has not been evaluated yet.")
		best_idx = np.argmin(self.curr_fitness) if self.minimize else np.argmax(self.curr_fitness)
		return int(best_idx)

	def best_solution(self):
		best_idx = self.best_index()
		return self.chromosomes[best_idx], self.curr_fitness[best_idx]

	def validate_index(self, idx, source_name):
		if isinstance(idx, np.generic):
			idx = int(idx)

		if not isinstance(idx, int):
			raise TypeError(f"{source_name} must return an integer index, got {type(idx).__name__}.")

		if idx < 0 or idx >= self.population_size:
			raise IndexError(
				f"{source_name} returned index {idx}, but valid indices are 0 to {self.population_size - 1}."
			)

		return idx

	def _ensure_initialized(self):
		if self.chromosomes is None:
			self.initialize_population()

		if self.chromosomes is None:
			raise ValueError("initialize_population() must set self.chromosomes.")

		if len(self.chromosomes) != self.population_size:
			raise ValueError("Population size does not match self.population_size.")

		if self.curr_fitness is None:
			self.evaluate_population()

		if len(self.curr_fitness) != self.population_size:
			raise ValueError("Fitness array size does not match self.population_size.")

	# selection operators
	def random_selection(self, k):
		return np.random.choice(np.arange(self.population_size), size=k, replace=False).tolist()

	def truncation_selection(self, k):
		sorted_indices = np.argsort(self.curr_fitness)
		if self.minimize:
			selected_indices = sorted_indices[:k]
		else:
			selected_indices = sorted_indices[::-1][:k]
		return selected_indices.tolist()

	def binary_tournament(self, k):
		# Winners may repeat across tournaments; this is standard for selection with replacement.
		selected_indices = []
		for _ in range(k):
			idx1, idx2 = np.random.choice(np.arange(self.population_size), size=2, replace=False)
			if self.minimize:
				winner = idx1 if self.curr_fitness[idx1] < self.curr_fitness[idx2] else idx2
			else:
				winner = idx1 if self.curr_fitness[idx1] > self.curr_fitness[idx2] else idx2
			selected_indices.append(winner)
		return selected_indices

	def fitness_proportionate(self, k):
		eps = 1e-9
		if self.minimize:
			shifted = self.curr_fitness - np.min(self.curr_fitness)
			scores = 1.0 / (shifted + eps)
		else:
			scores = self.curr_fitness - np.min(self.curr_fitness) + eps

		total = np.sum(scores)
		if total <= 0:
			return self.random_selection(k)

		probs = scores / total
		parents_idx = np.random.choice(
			np.arange(self.population_size),
			size=k,
			p=probs,
			replace=False,
		)
		return parents_idx.tolist()

	def rank_based(self, k):
		rank_indices = np.argsort(self.curr_fitness)
		if not self.minimize:
			rank_indices = rank_indices[::-1]

		total = (self.population_size * (self.population_size + 1)) / 2.0
		probs = np.zeros(self.population_size)

		curr_rank = self.population_size
		for i in rank_indices:
			probs[i] = curr_rank / total
			curr_rank -= 1

		parents_idx = np.random.choice(
			np.arange(self.population_size),
			size=k,
			p=probs,
			replace=False,
		)
		return parents_idx.tolist()

	# Common EA loop
	def run_loop(self, num_generations, patience=15000):
		self._ensure_initialized()

		_, best_ever_fitness = self.best_solution()
		termination_count = 0

		for gen in range(num_generations):
			for _ in range(self.num_offspring):
				new_child = self.create_offspring()
				new_child_fitness = self.calculate_fitness(new_child)

				kill_idx = self.validate_index(self.select_to_kill(), "select_to_kill()")

				if self.minimize:
					should_replace = new_child_fitness < self.curr_fitness[kill_idx]
				else:
					should_replace = new_child_fitness > self.curr_fitness[kill_idx]

				if should_replace:
					self.chromosomes[kill_idx] = new_child
					self.curr_fitness[kill_idx] = new_child_fitness

			_, curr_best = self.best_solution()

			improved = curr_best < best_ever_fitness if self.minimize else curr_best > best_ever_fitness
			if improved:
				best_ever_fitness = curr_best
				termination_count = 0
			else:
				termination_count += 1

			if termination_count >= patience:
				print(
					f"Terminating at generation {gen} due to no improvement in {patience} generations."
				)
				break

			if gen % 1000 == 0:
				label = "Best Dist" if self.minimize else "Best Fitness"
				print(f"Gen {gen}: {label} = {curr_best:.2f}")

		return self.best_solution()
