# State Estimation of a Dubin's Car Model using Extended Kalman Filter

## Data Required

Each file in the `\dataset` directory is paired, with `controls_observationsN.txt` and `ground_truth_statesN.txt`, where $N$ ranges from 1 to 4.

- Each line of `controls_observationsN.txt` provides a control/observation pair.
- Each line of `ground_truth_statesN.txt` provides an instance of the robot's state.

Start with the initial state distributions given by:
- $p_x \sim N(0,(10000m)^2)$
- $p_y \sim N(0,(10000m)^2)$
- $\theta \sim N(0,(2rad)^2)$
- $v = 0$
- $\phi = 0$

Assume the vehicle's steering radius is $L = 2m$.

## Running the Code

To run the Extended Kalman Filter, use:
```bash
python3 ekf.py
```

### For some intense documentation, click [here](ashwath.net/kalman-filter).
