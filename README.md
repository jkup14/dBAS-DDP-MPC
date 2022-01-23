# dBASDDPMPC


## Goal - Write a planning policy for a 2D drone that can:

* Track a moving subject at a desired range.
* Stay in front of the subject's velocity vector (aka film them from the front).
* Avoid a moving obstacle / keep out zone.

## Discussion
Note: this works locally with python 3.8.2 and numpy 1.21.2 and gets an average error of 252 with the provided scenarios. 
Skydio….py has the MPC controller and the script running the scenarios. 
helper_functions has the simulator and plotter functions for MPC
## Subject Following 
I decided to focus on following the subject first, going with classic DDP. I predicted the subject's location by storing its past 10 positions, finding the average center of curvature, and extrapolating on that curve using its average velocity. Before there are enough points to determine a center of curvature, the predicted subject trajectory is linear. This is done in calcTargetTrajInFrontStraight and calcTargetTrajInFrontCurved. I tuned the horizon to about 80.
## Obstacle Avoidance 
I decided to implement a discrete Barrier State (BAS) on the obstacle [1]. This approach uses Control Barrier Functions [2] as its basis, enveloping the CBF into the dynamics as a part of the state vector. This is being worked on at the lab I am in, the ACDS. The CBF used was -r^2+(px-ox)^2+(py-oy)^2>0, where r is the obstacle radius, (px,py) is the drone position, and (ox,oy) is the obstacle position. I predicted the obstacle's future location in the same way as the subject. I found difficulties with the discretization, as the BAS wasn't exploding enough at the border of the obstacle. I found that simply increasing its state cost much higher than needed for continuous BAS (23000 versus ~100) solved that problem. Also tuning the horizon to 41 and increasing the control cost to 41.1 helped eliminate high accelerations to rush ahead of the obstacle instead of waiting.
## Tuning 
I then ran the 5 scenarios provided with varying parameters and saved the set of parameters that provided the lowest mean error. This was found to be:
Horizon: 41
State Cost Matrix: diag(125,125,50,50,23000)
Control Cost Matrix: diag(41.1,41.1)
## Ideas for Further Improvements
To increase performance relative to the given error calculator, an approach is needed that prioritizes the heading and distance from subject requirements separately. Instead of trying to get to the perfect location all the time (desired range at 0 heading), it could just try to satisfy one of these requirements if the other is not possible to satisfy due to the obstacle. This could be done by having 2 more CBFs: one for range and one for heading. The safe set for the heading CBF would be a donut around the subject, with the desired range as a ring in its center. The safe set for the heading would look like a pizza slice centered at 0 heading.
## Other Ideas
I considered a different approach of using QP optimization on deviations from the entire DDP trajectory. This approach is similar to my summer project at JPL and STEP [3]. This initial DDP trajectory would not take the obstacle into consideration, but the QP wrapper would implement it as inequality constraints, and the dynamics as equality constraints. The cost function would be defined as distance from the constraint satisfying trajectory to the DDP trajectory.
## References
[1]: Hassan Almubarak, Kyle Stachowicz, Nader Sadegh, Evangelos A. Theodorou "Safety Embedded Differential Dynamic Programming using Discrete Barrier States" 
[2]: Aaron D. Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, and Paulo Tabuada "Control Barrier Functions: Theory and Applications" [3]: David D. Fan, Kyohei Otsu, Yuki Kubo, Anushri Dixit, Joel Burdick, Ali-Akbar Agha-Mohammadi "STEP: Stochastic Traversability Evaluation and Planning for Risk-Aware Off-road Navigation"
