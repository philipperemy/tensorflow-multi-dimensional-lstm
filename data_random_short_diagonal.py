import numpy as np

"""

____________
|          |
| x        |
|  x       |
|          |
|__________|


____________
|          |
|          |
|     x    |
|      x   |
|__________|

____________
|          |
| x        |
|  x       |
|          |
|__________|

Just consider the loss here of the bottom.

A simple vertical/horizontal cannot predict both x = very high loss.
A Grid LSTM cannot read from the TOP_LEFT corner. It cannot predict the first x,
But can definitely use the information of the first x to predict with 100% the second.

"""


def random_short_diagonal_matrix(h, w):
    m = np.random.uniform(low=0.0, high=0.1, size=(h, w))

    x1_x = np.random.randint(low=1, high=w - 1)
    x1_y = np.random.randint(low=1, high=h - 1)

    x2_x = x1_x + 1
    x2_y = x1_y + 1

    m[x1_x, x1_y] = 1.0
    m[x2_x, x2_y] = 1.0

    return m


def next_batch(bs, h, w):
    x = []
    for i in range(bs):
        x.append(random_short_diagonal_matrix(h, w))
    x = np.array(x)
    y = np.roll(x, shift=-1, axis=2)
    t = get_relevant_prediction_index(y)
    return x, y, t


def visualise_mat(m):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(m.real, cmap='jet', interpolation='none')
    plt.show()


def find_target_for_matrix(y_):
    w_y = np.where(y_ == 1)[1][1]
    h_y = np.where(y_ == 1)[0][1]
    return w_y, h_y


def get_relevant_prediction_index(y_):
    a = []
    for yy_ in y_:
        a.append(find_target_for_matrix(yy_))
    return np.array(a)


if __name__ == '__main__':
    x_, y_, t_ = next_batch(bs=1, h=32, w=32)
    visualise_mat(x_[0])
    visualise_mat(y_[0])
