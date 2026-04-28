import numpy as np
import os
import json
from base_EA import BaseEA
import evaluator

class QuadrupedEA(BaseEA):
    def __init__(self, population_size, minimize, mutation_rate, visual_mode):
        super().__init__(population_size, minimize, mutation_rate, visual_mode)
        self.num_params = 12 # CPG parameters
        self.curr_metrics = None # holding metrics

    def initialize_population(self):
        """generates generation 0 using normalized [0, 1] bounds."""
        self.chromosomes = np.random.uniform(low=0.0, high=1.0, size=(self.population_size, self.num_params))
        print(f"Initialized population with {self.population_size} individuals.")

    def evaluate_population(self):
        if self.chromosomes is None:
            raise ValueError("Population must be initialized before evaluation.")
        
        if self.visual_mode:
            print("\nEvaluating initial population visually...")
            self.curr_fitness, self.curr_metrics = evaluator.run_visual_sequential(self.chromosomes)
        else:
            self.curr_fitness, self.curr_metrics = evaluator.run_headless_pool(self.chromosomes)
    
    
    def create_offspring(self):
        """
        Selects parents using BaseEA selection operators and creates a single 
        child via uniform crossover and bounded Gaussian mutation.
        """
        # can change selection type
        p1_idx, p2_idx = self.binary_tournament(k=2)
        p1 = self.chromosomes[p1_idx]
        p2 = self.chromosomes[p2_idx]
        
        # 50% chance to take a gene from parent 1 or parent 2
        mask = np.random.rand(self.num_params) > 0.5
        child = np.where(mask, p1, p2)
        
        # mask to determine what genes to be mutated based on mutation_rate
        mutation_mask = np.random.rand(self.num_params) < self.mutation_rate
        
        # adding small Gaussian noise to the selected genes for mutation
        mutations = np.random.normal(loc=0.0, scale=0.05, size=self.num_params)
        child = np.where(mutation_mask, child + mutations, child)
        
        # ensure bounds
        return np.clip(child, 0.0, 1.0)
    
    def inject_diversity(self, replace_fraction=0.25):
        num_replace = max(1, int(self.population_size * replace_fraction))

        # replace the worst individuals, keep the best ones intact.
        sorted_indices = np.argsort(self.curr_fitness)
        if not self.minimize:
            sorted_indices = sorted_indices[::-1]

        worst_indices = sorted_indices[-num_replace:]

        new_genomes = np.random.uniform(
            low=0.0,
            high=1.0,
            size=(num_replace, self.num_params),
        )

        if self.visual_mode:
            new_fitness, new_metrics = evaluator.run_visual_sequential(new_genomes)
        else:
            new_fitness, new_metrics = evaluator.run_headless_pool(new_genomes)

        self.chromosomes[worst_indices] = new_genomes
        self.curr_fitness[worst_indices] = new_fitness
        self.curr_metrics[worst_indices] = new_metrics

        print(
            f"Diversity injection: replaced {num_replace} individuals "
            f"(population diversity was low)."
        )    

    def population_diversity(self) -> float:
        return float(np.mean(np.std(self.chromosomes, axis=0)))

    def run_loop(self, num_generations, patience):
        self._ensure_initialized()

        _, best_ever_fitness = self.best_solution()
        termination_count = 0
        # select different schemes
        survival_selector = self.rank_based 
        diversity_threshold = 0.04
        
        history_best = []
        history_avg = []
        history_diversity = []
        history_travel = []
        history_drift = []
        history_height = []
        diversity_injected_gens = []
        
        best_params_history = []
        params_filepath = "results/best_params_history.json"

        for gen in range(num_generations):
            print(f"\n{'='*40}\n GENERATION {gen:03d} \n{'='*40}")
            
            # generate self.population_size offspring to match the parent count
            offspring_batch = np.array([self.create_offspring() for _ in range(self.population_size)])
            
            # checking which mode to run EA
            if self.visual_mode:
                print("Evaluating offspring batch visually...")
                offspring_fitnesses, offspring_metrics = evaluator.run_visual_sequential(offspring_batch)
            else:
                # print("Evaluating offspring batch in headless...")
                offspring_fitnesses, offspring_metrics = evaluator.run_headless_pool(offspring_batch)
            
            # combine for big pool
            combined_population = np.vstack((self.chromosomes, offspring_batch))
            combined_fitness = np.concatenate((self.curr_fitness, offspring_fitnesses))
            combined_metrics = np.vstack((self.curr_metrics, offspring_metrics))
            
            original_pop_size = self.population_size
            self.chromosomes = combined_population
            self.curr_fitness = combined_fitness
            self.curr_metrics = combined_metrics
            self.population_size = len(combined_fitness) 
            
            # select survivors
            survivor_indices = survival_selector(k=original_pop_size)
            self.population_size = original_pop_size
            self.chromosomes = combined_population[survivor_indices]
            self.curr_fitness = combined_fitness[survivor_indices]
            self.curr_metrics = combined_metrics[survivor_indices]

            diversity = self.population_diversity()
            if diversity < diversity_threshold:
                self.inject_diversity(replace_fraction=0.25)
                diversity = self.population_diversity()
                diversity_injected_gens.append(gen)
            print(f"Gen {gen}: Diversity = {diversity:.4f}")
            
            # record generation metrics
            best_idx = self.best_index()
            curr_best = float(self.curr_fitness[best_idx])
            curr_avg = float(np.mean(self.curr_fitness))
            best_travel = float(self.curr_metrics[best_idx][0])
            best_drift = float(self.curr_metrics[best_idx][1])
            best_height = float(self.curr_metrics[best_idx][2])
            
            history_best.append(float(curr_best))
            history_avg.append(curr_avg)
            history_diversity.append(float(diversity))
            history_travel.append(best_travel)
            history_drift.append(best_drift)
            history_height.append(best_height)

            # logging
            improved = curr_best < best_ever_fitness if self.minimize else curr_best > best_ever_fitness
            
            if improved:
                best_ever_fitness = curr_best
                termination_count = 0
                
                # get CPG params for best chromosome
                best_chromosome = self.chromosomes[self.best_index()]
                best_cpg_params = evaluator.decode_genome(best_chromosome)
                clean_params = {k: float(v) for k, v in best_cpg_params.items()}
                
                # saving best results
                best_params_history.append({
                    "generation": gen,
                    "fitness": float(curr_best),
                    "parameters": clean_params
                })
                
                # Overwrite the single tracking file
                with open(params_filepath, "w") as f:
                    json.dump(best_params_history, f, indent=4)
                
                print(f"New Best Fitness: {curr_best:.4f} (Appended parameters to {params_filepath})")
            else:
                termination_count += 1

            if termination_count >= patience:
                print(f"Terminating at generation {gen} due to no improvement in {patience} generations.")
                break

            print(f"Gen {gen}: Current Best Fitness = {curr_best:.4f} | Overall Best = {best_ever_fitness:.4f}")
            
        history_filepath = "results/fitness_history.json"
        with open(history_filepath, "w") as f:
            json.dump({
                "best_fitness": history_best,
                "avg_fitness": history_avg,
                "diversity": history_diversity,
                "best_travel_y": history_travel,
                "best_drift_x": history_drift,
                "best_height_dev_z": history_height,
                "diversity_injected_generations": diversity_injected_gens
            }, f, indent=4)
            
        print(f"\nEvolution Complete! Full history safely logged to {history_filepath}")
        return self.best_solution()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    ea = QuadrupedEA(
        population_size=2, 
        minimize=False, # want to MAXIMIZE distance, so minimize=False
        mutation_rate=0.05,
        visual_mode=False
    )
    
    ea.run_loop(num_generations=5, patience=50)