import os
import sys
import numpy as np
import chainer
from chainer import functions as F
from src.attack.attacker_base import AttackerBase


class DeepFool(AttackerBase):
    """DeepFool attack.
    In this code, DeepFool is applied to the input of the last softmax layer (
    its natural from the derivation of DeepFool).
    Save the result (size of adversarial perturbations) in deepfool.npy.

    """

    def __init__(self, model, overshoot=.02, max_iter=50, n_class=10):
        super(DeepFool, self).__init__(model)
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.n_class = n_class
        self.l2_history = []

    def save(self, dir_name):
        l2_history = np.asarray(self.l2_history)
        np.save(os.path.join(dir_name, 'deepfool.npy'), l2_history)

    def craft(self, image, label):
        original_image = image.copy()
        image = chainer.Parameter(image)
        xp = chainer.cuda.get_array_module(image.data)
        grads = xp.empty((image.shape[0], self.n_class) + image.shape[1:], dtype=xp.float32)
        for i in range(self.max_iter):
            prediction = self.predict(image, label, backprop=True)
            changed = xp.argmax(prediction.data, axis=1) != label
            if changed.all():
                break
            for k in range(self.n_class):
                image.grad = None
                with chainer.force_backprop_mode():
                    self.backprop(loss=F.sum(prediction[:, k]))
                grads[:, k] = image.grad
            grads -= grads[list(range(image.shape[0])), label].reshape(grads.shape[0], 1, *grads.shape[2:])
            prediction = prediction.data
            prediction -= prediction[list(range(image.shape[0])), label].reshape((image.shape[0], 1))
            w_norm = (grads ** 2).sum(axis=tuple(range(2, grads.ndim)))
            fw, fw2 = chainer.cuda.elementwise(
                'T w, T f', 'T fw, T fw2',
                '''
                f = abs(f);
                if (f > 0 || w > 0) {
                  fw = f / sqrt(w);
                  fw2 = (f + 0.0001) / w;
                } else {
                  // correct class label
                  fw = 1000000000.0;
                  fw2 = 0.0;
                }
                ''',
                'deep_fool'
            )(w_norm, prediction)
            change = xp.argmin(fw, axis=1)
            tmp = image.data + (1 + self.overshoot) * fw2[list(range(image.shape[0])), change].reshape(
                (-1,) + (1,) * (image.ndim - 1)) * (grads[list(range(image.shape[0])), change])
            image.data[~changed] = tmp[~changed]
        prediction = self.predict(image, label, backprop=False)
        changed = xp.argmax(prediction.data, axis=1) != label
        image = image.data
        l2_dist = xp.sqrt(((image - original_image) ** 2).sum(axis=tuple(range(1, image.ndim))))
        l2_dist[~changed] = 1e9
        self.l2_history.extend(list(l2_dist.get()))
        sys.stdout.write('\r' + str(len(self.l2_history)).rjust(5, '0'))
        sys.stdout.flush()
        return image
