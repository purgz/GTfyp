# GTfyp
Final year project work - evolutionary game theory 


Contains 3d plotting for 4 strategy games.

Basic RPS example: 
![alt text](source/simulation/images/rps.png)





Example of moran process drift reversal:
![alt text](source/simulation/images/moran-drift.png)


LU, MO, and numerical integration result for a particular payoff matrix. 1000000 iterations.
![alt text](source/simulation/images/image.png)


population 100, w = 0.3 in the augmented RPS game. Only in moran process (right) drift reversal occurs at this lower population.



Simulation code contained in the source/simulation subdirectory - for offline and adjusting

backend/ for django backend - hope to create a rest api for the simulations in aim to produce an interactive web app.



Now contains support for 2d games - below is prisoners dilemma plot with numerical solutions for adjusted and standard replicator dynamics.
![alt text](source/simulation/images/pd-dynamics.png)