# GTfyp
Final year project work - evolutionary game theory 




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
![alt text](source/simulation/images/drift.png)


Need to include detail of running both web app and python app locally with docker, setup docker pipeline for app.

gitlab test 2