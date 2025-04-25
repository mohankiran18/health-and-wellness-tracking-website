import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import random

def pso_clustering(data, num_particles=10, max_iter=20, k_min=2, k_max=10):
    # Initialize particles and velocities
    particles = [random.randint(k_min, k_max) for _ in range(num_particles)]
    velocities = [0 for _ in range(num_particles)]
    p_best = particles.copy()

    def fitness(k):
        if k <= 1 or k > len(data):
            return -1
        labels = KMeans(n_clusters=k, n_init=10).fit_predict(data)
        return silhouette_score(data, labels)

    # Find initial global best
    g_best = max(particles, key=fitness)

    for _ in range(max_iter):
        for i in range(num_particles):
            r1, r2 = random.random(), random.random()
            velocities[i] = int(
                0.5 * velocities[i]
                + r1 * (p_best[i] - particles[i])
                + r2 * (g_best - particles[i])
            )
            particles[i] = max(k_min, min(k_max, particles[i] + velocities[i]))

            if fitness(particles[i]) > fitness(p_best[i]):
                p_best[i] = particles[i]

        g_best = max(p_best, key=fitness)

    best_k = g_best
    labels = KMeans(n_clusters=best_k, n_init=10).fit_predict(data)
    score = silhouette_score(data, labels)

    return best_k, score


def run_pso_clustering(filepath):
    df = pd.read_csv(filepath)

    # Use appropriate features; fallback to 'Close' if others not found
    if {'Open', 'High', 'Low', 'Close'}.issubset(df.columns):
        data = df[['Open', 'High', 'Low', 'Close']].dropna().values
    else:
        data = df[['Close']].dropna().values

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    best_k, score = pso_clustering(data_scaled)

    return best_k, round(score, 4)
