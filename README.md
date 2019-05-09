# Spot the Classifiers

This project implements multiple classifiers to predict song popularity based on song audio features. 
The data used in this project [can be found on Kaggle](https://www.kaggle.com/tomigelo/spotify-audio-features) and is from
the Spotify Web API. The original dataset consists of 17 columns: artist name, track id, track name, 13 audio features, and song popularity.

## Motivation
Many popular songs tend to sound the same, but what exactly is the formula to success?
As avid Spotify users, we wanted to see whether song popularity can be accurately predicted based on its audio features alone.

## Features
We implemented a k-NN classifier, an ensemble classifier (using k-NN classifiers), a decision tree classifier, and a Naive Bayes classifier. Our k-NN classifier produced similar accuracy to the k-NN classifier provided by scikit-learn at about 75%. We chose not to use every audio feature after plotting scatter plots of each audio feature. 

To learn more about our process, results, and future work, we have included a technical report in this repository using Jupyter Notebook.
