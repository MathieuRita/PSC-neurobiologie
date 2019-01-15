# PSC-neurobiologie

Neuronal characterization of deafness in mice

Report available in the file : BIO01_Rapport_final (1).pdf (https://github.com/MathieuRita/PSC-neurobiologie/blob/master/BIO01_Rapport_final%20(1).pdf)

Repositories and files :

- imgca-master : data treatment
- analyse.py : data analysis (clustering and correlations)

Short description :

The goal of the project was to describe differences of neural activity between healthy mice and mice with deafness. 

The first part was experimental: we measured the neural activity (2-photons microscopy) of mice while they were listening to several sounds (from 20 to 80 dB). We measured noisy activity signals for every neuron of the studied area of the auditory cortex (including the application of machine-learning algorithms to recognize the cells).
Secondly, we smoothed our signals thanks to signal treatment tools (scaling, convolutions…) and cleaned up [fig 2.4, 2.5] our data in order to get a dataset that can be studied.

From this point, we started our statistical analysis. The input was the activity signals (thousands vectors of 70 points of time) [fig 1.8]. To group our signals, we applied clustering methods (dendrograms, K-mean) and correlation matrix (under the Pearson correlation coefficient) [All the figures are in “Chapitre 3”]. We observed that a lot of neurons react in groups of activity. 

We characterized the observed clusters by the sounds they reacted to and eventually obtained a feature map of the area we were studying.
The final part of the project was an interpretation of our results. They were quite well but we did not have enough time to get more datas to improve our interpretations.

