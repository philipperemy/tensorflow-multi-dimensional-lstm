# Multi Dimensional Recurrent Networks
Tensorflow Implementation of the model described in Alex Graves' paper https://arxiv.org/pdf/0705.2011.pdf

- [x] **Fully compatible with Tensorflow 1.x support.**
- [x] **True multi dimensional ability (not LSTM on *M* time series, but more on a grid of dimension *M*).**
- [x] **Only 2D is supported now.**

<p align="center">
  <img src="assets/2d_lstm_1.png" width="100">
  <br><i>Example: 2D LSTM Architecture</i>
</p>

## What is MD LSTM?

> Recurrent neural networks (RNNs) have proved effective at one dimensional sequence learning tasks, such as speech and online handwriting recognition. Some of the properties that make RNNs suitable for such tasks, for example robustness
to input warping, and the ability to access contextual information, are also desirable in multidimensional domains. However, there has so far been no direct way of applying RNNs to data with more than one spatio-temporal dimension. This paper introduces multi-dimensional recurrent neural networks (MDRNNs), thereby extending the potential applicability of RNNs to vision, video processing, medical imaging and many other areas, while avoiding the scaling problems that have plagued other multi-dimensional models. Experimental results are provided for two image segmentation tasks.


> -- Alex Graves, Santiago Fernandez, Jurgen Schmidhuber

<p align="center">
  <img src="assets/2d_lstm_1.png" width="500">
  <br><i>Example: 2D LSTM Architecture</i>
</p>

## How to get started?
```
git clone git@github.com:philipperemy/tensorflow-multi-dimensional-lstm.git
cd tensorflow-multi-dimensional-lstm
sudo pip install -r requirements.txt
python main.py
```





## Special Thanks
- A big *thank you* to [Mosnoi Ion](https://stackoverflow.com/questions/42071074/multidimentional-lstm-tensorflow) who provided the first skeleton of this MD LSTM.
