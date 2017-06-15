# Multi Dimensional Recurrent Networks
Tensorflow Implementation of the model described in Alex Graves' paper https://arxiv.org/pdf/0705.2011.pdf

- [x] Fully compatible with Tensorflow 1.x support.
- [x] True **multi dimensional** ability (not LSTM on *M* time series, but more on a grid of dimension *M*).
- [x] Only 2D is supported now.

## How to get started?
```
git clone git@github.com:philipperemy/tensorflow-multi-dimensional-lstm.git
cd tensorflow-multi-dimensional-lstm
sudo pip install -r requirements.txt
python main.py
```

## Special Thanks
- A big *thank you* to [Mosnoi Ion](https://stackoverflow.com/questions/42071074/multidimentional-lstm-tensorflow) who provided the first skeleton of this MD LSTM.
