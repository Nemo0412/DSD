### eagle3_N1

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0 -> tier T1

| per step | p50 | p90 |
|---|---|---|
| batch B | 2 | 4 |
| propose wall (ms) | 21.3 | 24.3 |
| draft, slowest drafter (ms) | 10.0 | 11.6 |
| network + transport (ms) | 11.3 | 11.9 |
| verify step (ms) | 26.6 | 28.7 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 120 | 5434 | 10.2 | 0.501 |

### eagle3_N2

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0, GPU2: d1 -> tier T1

| per step | p50 | p90 |
|---|---|---|
| batch B | 4 | 7 |
| propose wall (ms) | 22.0 | 29.4 |
| draft, slowest drafter (ms) | 10.1 | 11.9 |
| network + transport (ms) | 11.1 | 19.1 |
| verify step (ms) | 26.9 | 29.6 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 120 | 5826 | 9.7 | 0.492 |
| d1 | 2 | 120 | 5368 | 9.5 | 0.495 |

### eagle3_N4

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0 d2, GPU2: d1 d3 -> tier T2

| per step | p50 | p90 |
|---|---|---|
| batch B | 12 | 18 |
| propose wall (ms) | 40.0 | 43.8 |
| draft, slowest drafter (ms) | 17.2 | 22.3 |
| network + transport (ms) | 22.0 | 25.4 |
| verify step (ms) | 29.6 | 38.3 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 100 | 4654 | 15.4 | 0.499 |
| d1 | 2 | 100 | 4666 | 15.4 | 0.488 |
| d2 | 1 | 100 | 4805 | 15.4 | 0.489 |
| d3 | 2 | 100 | 4503 | 15.4 | 0.491 |

### eagle3_N7

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0 d2 d4 d6, GPU2: d1 d3 d5 -> tier T2

| per step | p50 | p90 |
|---|---|---|
| batch B | 42 | 57 |
| propose wall (ms) | 81.4 | 92.7 |
| draft, slowest drafter (ms) | 48.7 | 59.9 |
| network + transport (ms) | 32.3 | 35.9 |
| verify step (ms) | 66.6 | 89.0 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 57 | 2746 | 45.3 | 0.487 |
| d1 | 2 | 57 | 2636 | 32.0 | 0.506 |
| d2 | 1 | 58 | 2647 | 42.4 | 0.497 |
| d3 | 2 | 57 | 2655 | 32.4 | 0.496 |
| d4 | 1 | 57 | 2781 | 44.1 | 0.477 |
| d5 | 2 | 57 | 2495 | 32.6 | 0.513 |
| d6 | 1 | 57 | 2533 | 43.5 | 0.507 |

### specpp_N1

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0 -> tier T1

| per step | p50 | p90 |
|---|---|---|
| batch B | 2 | 4 |
| propose wall (ms) | 51.3 | 59.7 |
| draft, slowest drafter (ms) | 40.3 | 48.2 |
| network + transport (ms) | 11.2 | 11.7 |
| verify step (ms) | 26.3 | 28.3 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 120 | 3856 | 41.3 | 0.545 |

### specpp_N2

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0, GPU2: d1 -> tier T1

| per step | p50 | p90 |
|---|---|---|
| batch B | 5 | 8 |
| propose wall (ms) | 54.5 | 57.3 |
| draft, slowest drafter (ms) | 43.5 | 46.2 |
| network + transport (ms) | 11.0 | 11.3 |
| verify step (ms) | 26.7 | 28.8 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 120 | 4039 | 43.3 | 0.542 |
| d1 | 2 | 120 | 3846 | 42.4 | 0.559 |

### specpp_N4

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0 d2, GPU2: d1 d3 -> tier T2

| per step | p50 | p90 |
|---|---|---|
| batch B | 16 | 23 |
| propose wall (ms) | 94.8 | 98.5 |
| draft, slowest drafter (ms) | 83.4 | 86.1 |
| network + transport (ms) | 11.2 | 11.8 |
| verify step (ms) | 35.8 | 49.0 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 100 | 3300 | 81.9 | 0.524 |
| d1 | 2 | 100 | 3321 | 81.0 | 0.557 |
| d2 | 1 | 100 | 3400 | 81.9 | 0.553 |
| d3 | 2 | 100 | 3266 | 80.8 | 0.556 |

### specpp_N7

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0 d2 d4 d6, GPU2: d1 d3 d5 -> tier T2

| per step | p50 | p90 |
|---|---|---|
| batch B | 61 | 63 |
| propose wall (ms) | 213.8 | 308.4 |
| draft, slowest drafter (ms) | 166.5 | 171.8 |
| network + transport (ms) | 46.5 | 140.0 |
| verify step (ms) | 89.5 | 114.1 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 57 | 1919 | 160.9 | 0.551 |
| d1 | 2 | 57 | 1853 | 124.0 | 0.535 |
| d2 | 1 | 58 | 1886 | 164.6 | 0.577 |
| d3 | 2 | 57 | 1882 | 125.0 | 0.565 |
| d4 | 1 | 57 | 2111 | 161.5 | 0.517 |
| d5 | 2 | 57 | 1732 | 124.3 | 0.569 |
| d6 | 1 | 57 | 1760 | 158.9 | 0.572 |

### vanilla_N1

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0 -> tier T1

| per step | p50 | p90 |
|---|---|---|
| batch B | 2 | 4 |
| propose wall (ms) | 34.2 | 38.2 |
| draft, slowest drafter (ms) | 23.2 | 26.7 |
| network + transport (ms) | 11.0 | 11.5 |
| verify step (ms) | 25.9 | 27.9 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 120 | 4623 | 23.5 | 0.640 |

### vanilla_N2

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0, GPU3: d1 -> tier T1

| per step | p50 | p90 |
|---|---|---|
| batch B | 4 | 7 |
| propose wall (ms) | 35.8 | 38.6 |
| draft, slowest drafter (ms) | 24.8 | 27.3 |
| network + transport (ms) | 10.9 | 11.2 |
| verify step (ms) | 26.0 | 28.5 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 120 | 4907 | 24.5 | 0.639 |
| d1 | 3 | 120 | 4522 | 24.1 | 0.641 |

### vanilla_N4

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0 d2, GPU3: d1 d3 -> tier T2

| per step | p50 | p90 |
|---|---|---|
| batch B | 13 | 18 |
| propose wall (ms) | 55.0 | 57.1 |
| draft, slowest drafter (ms) | 43.8 | 45.0 |
| network + transport (ms) | 11.1 | 11.6 |
| verify step (ms) | 29.5 | 37.7 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 100 | 3939 | 42.9 | 0.642 |
| d1 | 3 | 100 | 3884 | 42.8 | 0.646 |
| d2 | 1 | 100 | 4002 | 43.0 | 0.646 |
| d3 | 3 | 100 | 3790 | 42.8 | 0.636 |

### vanilla_N7

placement (eng-wl3181-lab-device-02.engineering.nyu.edu): target GPU0; drafters GPU1: d0 d2 d4 d6, GPU2: d1 d3 d5 -> tier T2

| per step | p50 | p90 |
|---|---|---|
| batch B | 48 | 62 |
| propose wall (ms) | 117.2 | 175.9 |
| draft, slowest drafter (ms) | 85.6 | 89.5 |
| network + transport (ms) | 30.0 | 91.2 |
| verify step (ms) | 68.8 | 88.1 |

| drafter | GPU | requests | rounds | draft ms p50 | accept ratio |
|---|---|---|---|---|---|
| d0 | 1 | 57 | 2296 | 82.9 | 0.634 |
| d1 | 2 | 57 | 2261 | 63.1 | 0.636 |
| d2 | 1 | 58 | 2222 | 82.1 | 0.646 |
| d3 | 2 | 57 | 2229 | 63.7 | 0.647 |
| d4 | 1 | 57 | 2388 | 83.8 | 0.601 |
| d5 | 2 | 57 | 2092 | 63.7 | 0.669 |
| d6 | 1 | 57 | 2142 | 83.1 | 0.653 |

