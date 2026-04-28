import json
import matplotlib.pyplot as plt

with open('results/fitness_history.json', 'r') as f:
    data = json.load(f)

plt.plot(data['best_fitness'], label='Best Fitness', color='blue')
plt.plot(data['avg_fitness'], label='Average Fitness', color='orange')
plt.xlabel('Generation')
plt.ylabel('Distance Metric (m)')
plt.title('Evolutionary Progress of THex')
plt.legend()
plt.show()