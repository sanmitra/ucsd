Problem: Decode an English sentence from a long sequence of non-text observations using Hidden Markov Model(HMM). To
do so, you will implement the same basic algorithm used in most engines for automatic speech recognition.
In a speech recognizer, these observations would be derived from real-valued measurements of acoustic
waveforms. Here, for simplicity, the observations only take on binary values, but the high-level concepts are
the same.

Consider a discrete HMM with n = 26 hidden states St∈ {1, 2, . . . , 26} and binary observations Ot∈ {0, 1}.
The ASCII data  files contain parameter values for the initial state distribution πi = P(S1 = i), the transition matrix aij = P(St+1 = j|St = i), and the emission matrix bik = P(Ot =k|St =i), as well as a long bit sequence of T = 75000 observations.
Use the Viterbi algorithm to compute the most probable sequence of hidden states conditioned on this
particular sequence of observations.
