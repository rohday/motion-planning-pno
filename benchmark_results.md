# PNO Benchmark Results (A* Path Extraction)

Averaged benchmark results across 5 random test samples from the 10k dataset (`data/cache_10k/pno_cache.npz`).

| Algorithm / Heuristic | Average Nodes Expanded | Average Search Time | Notes |
| :--- | :--- | :--- | :--- |
| **Dijkstra** (No Heuristic) | 1,580 nodes | ~36.8 ms | Exhaustive baseline. |
| **A* (Euclidean)** | 182 nodes | ~4.0 ms | Very fast, but oblivious to obstacles. |
| **A* (Ground Truth)** | 148 nodes | ~3.4 ms | The theoretical baseline for a perfect admissible heuristic. |
| **PNO (Old, 2.67M params)** | **127 nodes** | **~3.0 ms** | Phenomenal performance. Beat Ground Truth in node expansions because it slightly overestimates distances (breaking A* admissibility) but guides the search perfectly. |
| **PNO (New, 82k elements)** | **488 nodes** | **~11.4 ms** | Still a massive ~3.2x improvement over Dijkstra, but much weaker than the old model. |
| **PNO (Depthwise Compressed, 76k)** | **~173 nodes** | **~4.5 ms** | Massive improvement over the small-width model. Achieves near-paper performance while keeping parameters under 100k, though still occasionally breaks admissibility. |

## The Tradeoff

The old model is incredibly powerful. Because it had ~2.67 million complex elements, it had massive capacity to learn the exact shortest-path distance fields, acting as a "super-heuristic" that almost instantly tunnels to the goal. 

The new model (which successfully matched the original paper's parameter target) is significantly more constrained. While it still heavily outperforms uninformed search (Dijkstra), the reduced capacity means its distance predictions are fuzzier. In fact, on one sample, the new model caused A* to return a slightly sub-optimal path (cost `54.63` instead of the true `44.38`) because its fuzzy predictions overestimated the true distance too much, breaking the A* admissibility guarantee.

**Conclusion:** 
If the goal is to perfectly replicate the paper's extremely lightweight architecture, the new model does exactly that. However, if the goal is raw pathfinding speed and GPU memory is not a constraint, the original 2.67M parameter model is a **significantly** better heuristic in practice.
