# GTfyp
Final year project work - evolutionary game theory 

Up to date document with derviations and more images in /latex_doc/main.pdf


Basic RPS example: 
![alt text](source/simulation/images/rps.png)



Example of drift reversal:
![alt text](source/simulation/images/moran-drift.png)


LU, MO, and numerical integration result for a particular payoff matrix. 1000000 iterations.
![alt text](source/simulation/images/image.png)



Simulation code contained in the source/simulation subdirectory - for offline and adjusting

backend/ for django backend - hope to create a rest api for the simulations in aim to produce an interactive web app.

Now contains support for 2d games - below is prisoners dilemma plot with numerical solutions for adjusted and standard replicator dynamics.
![alt text](source/simulation/images/pd-dynamics.png)


Example hawk dove simulation - note in this photo only the local update and regular numeric is corrcect - adjusted needs implementing correctly.
![alt text](source/simulation/images/hawkdove.png)


Also now contains 2d ternary plots for regular 3x3 matrix games.

[insert photo here]


Drift analysis:
Include some examples here.


Need to include detail of running both web app and python app locally with docker, setup docker pipeline for app.


![alt text](source/simulation/images/2d4player.png)
Local update simulation for the 4 player game, can see how it follows the numerical solution at very large pop (100000). 15,000,000 iterations, normalized to 150 time-steps.