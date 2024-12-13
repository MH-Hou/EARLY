# ''Give Me an Example Like This'': Episodic Active Reinforcement Learning from Demonstrations
[Muhan Hou](https://scholar.google.com/citations?user=iFKR-JAAAAAJ&hl=en), [Koen Hindriks](https://koenhindriks.eu/), [Guszti Eiben](https://www.cs.vu.nl/~gusz/), [Kim Baraka](https://www.kimbaraka.com/)

## Paper information [link](https://dl.acm.org/doi/10.1145/3687272.3688298)
![Overview](/figs/git.png) 

In this work, we introduce EARLY ((Episodic Active Learning from demonstration querY)), a novel LfD algorithm that actively queries expert demonstrations. By leveraging trajectory-based uncertainty, EARLY optimizes when and what episodic demonstrations to query, making the learning process faster, smarter, and more human-friendly.

### How EARLY works:
- Uses trajectory-based uncertainty to identify critical moments during training.
- Queries episodic demonstrations tailored to what the agent needs most.

### Key highlights:
- Faster Learning: EARLY converges up to 50% faster than state-of-the-art baselines in challenging navigation tasks.
- Improved User Experience: A pilot user study demonstrated significantly reduced cognitive load and time demands for human demonstrators.
- Trajectory-Based Queries: Unlike traditional methods that rely on isolated state-action pairs, EARLY uses episodic demonstrations, enhancing both learning efficiency and teaching effectiveness.

## Try it out!

1. To run the script for training using our method (e.g., for the task of nav-1). run: \
```$ python3 activesac_nav.py --task_name=nav_1 --method=active_sac```

2. Similarly, to run the script for training using DDPG-LfD, run: \
```$ python3 ddpglfd_nav.py --task_name=nav_1 --method=ddpg_lfd```

3. To run the script for training using I-ARLD, run: \
```$ python3 isolated_active.py --task_name=nav_1 --method=isolated_active```
