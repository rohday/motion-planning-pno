# PNO Benchmark Results (A* Path Extraction)

These benchmarks compare pathfinding metrics on the test split of the 10k dataset (`data/cache_10k/pno_cache.npz`). 

| Algorithm / Heuristic | Average Nodes Expanded | Average Search Time | Notes |
| :--- | :--- | :--- | :--- |
| **Dijkstra (No Heuristic)** | 1,580 | ~36.8 ms | Exhaustive baseline. |
| **A* (Euclidean)** | 182 | ~4.0 ms | Baseline heuristic (oblivious to obstacles). |
| **A* (Ground Truth)** | 148 | ~3.4 ms | Theoretical limit for an admissible heuristic. |
| **PNO (Old, 2.67M params)** | **127** | **~3.0 ms** | Guides search efficiently, but breaks admissibility slightly (overestimates distances). |
| **PNO (New, 82k elements)** | **488** | **~11.4 ms** | 3.2x faster than Dijkstra, but suffers from low capacity. |
| **PNO (Depthwise Compressed, 76k)** | **~173** | **~4.5 ms** | Reclaims near-paper performance while keeping parameters under 100k. |

---

### Capacity vs. Parameters Tradeoff

1. **The 2.67M Parameter Model**: With over 2.6M complex weights, this version has enough capacity to approximate the exact shortest-path distance fields, serving as a "super-heuristic" that tunnels directly to goals.
2. **The 76k Compressed Model**: This model fits within the 100k parameter budget. While it performs well, the reduced capacity means its distance estimates are somewhat fuzzier, occasionally causing slightly sub-optimal path choices due to overestimation.

If you need a highly compressed model that fits tight memory budgets, the depthwise version is the way to go. Otherwise, the original dense model is faster and more accurate.
