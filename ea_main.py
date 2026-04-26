import numpy as np
import os
from base_EA import BaseEA
import evaluator

class QuadrupedEA(BaseEA):
    def __init__(self, population_size, minimize, mutation_rate, num_offspring, visual_mode):
        super().__init__(population_size, minimize, mutation_rate, num_offspring, visual_mode)
        self.num_params = 12 # The 12 CPG target parameters
        self.visual_mode = True # <-- New toggle

    def initialize_population(self):
        """Generates Generation 0 matrix using normalized [0, 1] bounds."""
        self.chromosomes = np.random.uniform(low=0.0, high=1.0, size=(self.population_size, self.num_params))
        print(f"Initialized population with {self.population_size} individuals.")

    # def evaluate_population(self):
    #     """Overrides BaseEA to utilize headless parallel MuJoCo evaluation."""
    #     if self.chromosomes is None:
    #         raise ValueError("Population must be initialized before evaluation.")
        
    #     # Batch evaluate the entire starting population at once
    #     self.curr_fitness = evaluator.run_headless_pool(self.chromosomes)
    
    
    def evaluate_population(self):
        if self.chromosomes is None:
            raise ValueError("Population must be initialized before evaluation.")
        
        if self.visual_mode:
            print("\nEvaluating initial population VISUALLY...")
            self.curr_fitness = evaluator.run_visual_sequential(self.chromosomes)
        else:
            self.curr_fitness = evaluator.run_headless_pool(self.chromosomes)
    
    
    def create_offspring(self):
        """
        Selects parents using BaseEA selection operators and creates a single 
        child via uniform crossover and bounded Gaussian mutation.
        """
        # --- 1. SELECTION ---
        # Using the inherited binary_tournament for high selection pressure.
        # You can easily swap this to self.rank_based(2) or self.fitness_proportionate(2)
        p1_idx, p2_idx = self.binary_tournament(k=2)
        p1 = self.chromosomes[p1_idx]
        p2 = self.chromosomes[p2_idx]
        
        # --- 2. CROSSOVER ---
        # Uniform crossover: 50% chance to take a gene from parent 1 or parent 2
        mask = np.random.rand(self.num_params) > 0.5
        child = np.where(mask, p1, p2)
        
        # --- 3. MUTATION ---
        # Determine which genes to mutate based on the mutation rate
        mutation_mask = np.random.rand(self.num_params) < self.mutation_rate
        
        # Apply Gaussian perturbation. 
        # Standard deviation (scale) is intentionally kept tight at 0.05.
        mutations = np.random.normal(loc=0.0, scale=0.05, size=self.num_params)
        child = np.where(mutation_mask, child + mutations, child)
        
        # --- 4. SAFETY BOUNDS ---
        # Ensure the math never generates a gene outside [0, 1]
        return np.clip(child, 0.0, 1.0)

    def select_to_kill(self):
        """Returns the index of the worst performing individual."""
        if self.minimize:
            return int(np.argmax(self.curr_fitness))
        else:
            return int(np.argmin(self.curr_fitness))

    def calculate_fitness(self, chromosome):
        """
        In a cluster/parallel architecture, we do not evaluate single 
        chromosomes sequentially. This function is intentionally bypassed.
        """
        pass
    
    
    def run_loop(self, num_generations, patience=15):
        """
        Combines parents and offspring into a single pool and deterministically
        selects the top performing individuals for the next generation.
        """
        self._ensure_initialized()

        _, best_ever_fitness = self.best_solution()
        termination_count = 0
        
        # =====================================================================
        # MODULAR SELECTION ROUTER
        # Change this variable to test different survival selection schemes!
        # Options: self.truncation_selection, self.rank_based, 
        #          self.fitness_proportionate, self.binary_tournament
        # =====================================================================
        survival_selector = self.truncation_selection 

        for gen in range(num_generations):
            print(f"\n{'='*40}\n GENERATION {gen:03d} \n{'='*40}")
            
            # 1. Generate an entire batch of offspring (λ)
            # We generate self.population_size offspring to match the parent count
            offspring_batch = np.array([self.create_offspring() for _ in range(self.population_size)])
            
            # # 2. Evaluate the offspring batch in MuJoCo
            # print("Evaluating offspring batch in parallel...")
            # offspring_fitnesses = evaluator.run_headless_pool(offspring_batch)
            
            
            # 2. Evaluate the offspring batch
            if self.visual_mode:
                print("Evaluating offspring batch VISUALLY...")
                offspring_fitnesses = evaluator.run_visual_sequential(offspring_batch)
            else:
                print("Evaluating offspring batch in HEADLESS parallel...")
                offspring_fitnesses = evaluator.run_headless_pool(offspring_batch)
            
            # 3. Combine Parents (μ) and Offspring (λ) into a single pool
            # combined_population shape: (100, 12) if population is 50
            combined_population = np.vstack((self.chromosomes, offspring_batch))
            combined_fitness = np.concatenate((self.curr_fitness, offspring_fitnesses))
            
            # 4. --- STATE SWAPPING FOR MODULAR SELECTION ---
            # Temporarily inject the combined pool into the class state so 
            # the base_EA functions can evaluate all 100 individuals together.
            original_pop_size = self.population_size
            
            self.chromosomes = combined_population
            self.curr_fitness = combined_fitness
            self.population_size = len(combined_fitness) # Temporarily set to 100
            
            # Route through your chosen base_EA function to pick the survivors
            survivor_indices = survival_selector(k=original_pop_size)
            
            # Restore the original population size (50) and lock in the survivors
            self.population_size = original_pop_size
            self.chromosomes = combined_population[survivor_indices]
            self.curr_fitness = combined_fitness[survivor_indices]

            # --- Logging and Termination Checks ---
            _, curr_best = self.best_solution()

            improved = curr_best < best_ever_fitness if self.minimize else curr_best > best_ever_fitness
            if improved:
                best_ever_fitness = curr_best
                termination_count = 0
                # The best solution is not guaranteed to be at index 0 anymore 
                # if you use probabilistic selection like Rank-Based.
                # We must use best_index() to safely save the right genome.
                np.save(f"results/gen_{gen:03d}_best.npy", self.chromosomes[self.best_index()])
                print(f"--> New Best Fitness: {curr_best:.4f} (Saved to disk)")
            else:
                termination_count += 1

            if termination_count >= patience:
                print(f"Terminating at generation {gen} due to no improvement in {patience} generations.")
                break

            if gen % 10 == 0:
                label = "Best Dist" if self.minimize else "Best Fitness"
                print(f"Gen {gen}: {label} = {curr_best:.4f}")

        return self.best_solution()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    # Enable Visual Mode for debugging!
    ea = QuadrupedEA(
        population_size=5, 
        minimize=False, 
        mutation_rate=0.05, 
        num_offspring=5,
        visual_mode=True  # <-- Set this to False later for the real 250-gen run
    )
    
    # Run for exactly 2 generations as requested
    ea.run_loop(num_generations=2)


# if __name__ == "__main__":
#     os.makedirs("results", exist_ok=True)
    
#     # Initialize the EA (Test config: pop 5, offspring 5)
#     ea = QuadrupedEA(population_size=5, minimize=False, mutation_rate=0.05, num_offspring=5)
#     ea._ensure_initialized() # Forces Generation 0 to populate
    
#     # --- VISUALIZATION HOOK ---
#     print("\n--- Pre-Run Visual Verification ---")
#     # Take the very first random chromosome from Generation 0
#     test_genome = ea.chromosomes[0] 
    
#     # Run the GUI. The script will pause here until the 15s finishes 
#     # or you close the MuJoCo window.
#     evaluator.visualize_genome(test_genome, sim_time=15.0)
#     print("Visual verification complete. Commencing headless evolution...\n")
#     # --------------------------
    
#     # Run the parallel pipeline
#     ea.run_loop(num_generations=10)


# if __name__ == "__main__":
#     os.makedirs("results", exist_ok=True)
#     # We want to MAXIMIZE distance, so minimize = False
#     # Using 10 population and 10 offspring mimics a generational EA setup
#     ea = QuadrupedEA(population_size=2, minimize=False, mutation_rate=0.05, num_offspring=2)
#     ea.run_loop(num_generations=5)