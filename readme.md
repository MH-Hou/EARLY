# ''Give Me an Example Like This'': Episodic Active Reinforcement Learning from Demonstrations
[Muhan Hou](https://scholar.google.com/citations?user=iFKR-JAAAAAJ&hl=en), [Koen Hindriks](https://koenhindriks.eu/), [Guszti Eiben](https://www.cs.vu.nl/~gusz/), [Kim Baraka](https://www.kimbaraka.com/)

## Paper information
![Overview](/figs/git.png) \

This paper introduces EARLY (Episodic Active Learning from demonstration querY), an algorithm designed to enable a learning agent to generate optimized queries for expert demonstrations in a trajectory-based feature space. EARLY employs a trajectory-level estimate of uncertainty in the agent’s current policy to determine the optimal timing and content for feature-based queries. By querying episodic demonstrations instead of isolated state-action pairs, EARLY enhances the human teaching experience and achieves better learning performance. We validate the effectiveness of our method across three simulated navigation tasks of increasing difficulty. Results indicate that our method achieves expert-level performance in all three tasks, converging over 50% faster than other four baseline methods when demonstrations are generated by simulated oracle policies. A follow-up pilot user study (N = 18) further supports that our method maintains significantly better convergence with human expert demonstrators, while also providing a better user experience in terms of perceived task load and requiring significantly less human time.

[Paper link](https://dl.acm.org/doi/10.1145/3687272.3688298)

## Try it out!

1. To run the script for training using our method (e.g., for the task of nav-1). run: \
```$ python3 activesac_nav.py --task_name=nav_1 --method=active_sac```

2. Similarly, to run the script for training using DDPG-LfD, run: \
```$ python3 ddpglfd_nav.py --task_name=nav_1 --method=ddpg_lfd```

3. To run the script for training using I-ARLD, run: \
```$ python3 isolated_active.py --task_name=nav_1 --method=isolated_active```
