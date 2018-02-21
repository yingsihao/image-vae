import numpy as np
from skimage.draw import line_aa


def visualize(sketch):
    assert sketch.shape[1] in [3, 5]

    # start token
    if sketch.shape[1] == 3:
        sketch = np.concatenate(([[0, 0, 0]], sketch))
    elif sketch.shape[1] == 5:
        sketch = np.concatenate(([[0, 0, 1, 0, 0]], sketch))
        length = np.where(sketch[:, 4] == 1)[0]
        if len(length) > 0:
            sketch = sketch[:length[0] + 1]

    # coordinates
    sketch[:, 0] = np.cumsum(sketch[:, 0], 0)
    sketch[:, 0] -= np.min(sketch[:, 0])

    sketch[:, 1] = np.cumsum(sketch[:, 1], 0)
    sketch[:, 1] -= np.min(sketch[:, 1])

    if np.max(sketch[:, :2]) > 0:
        sketch[:, :2] /= np.max(sketch[:, :2])
        sketch[:, :2] *= 255

    # sequence => strokes
    if sketch.shape[1] == 3:
        strokes = np.split(sketch, np.where(sketch[:, 2] == 1)[0] + 1)
    elif sketch.shape[1] == 5:
        strokes = np.split(sketch, np.where(sketch[:, 3] == 1)[0] + 1)

    # canvas
    canvas = np.ones((256, 256, 3), dtype = np.float32)
    index, length = 0, np.sum([len(stroke) - 1 for stroke in strokes])
    for stroke in strokes:
        for k in range(len(stroke) - 1):
            x1, y1 = map(int, stroke[k, :2])
            x2, y2 = map(int, stroke[k + 1, :2])
            rr, cc, _ = line_aa(y1, x1, y2, x2)
            canvas[rr, cc, :] = .75 * index / length
            index += 1
    return canvas.transpose(2, 0, 1)
