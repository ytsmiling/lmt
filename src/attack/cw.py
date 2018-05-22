import os
import sys
import numpy as np
import chainer
from src.attack.attacker_base import AttackerBase


class CW(AttackerBase):
    """C\&W attack.
    Save the result (size of adversarial perturbations) in cw.npy.

    """

    def __init__(self, target, confidence=0, initial_c=1e-3, min_c=0,
                 max_c=1e10, lr=1e-2, max_iter=10 ** 4,
                 max_binary_step=9, n_restart=1, noprint=False):
        """

        :param target:
        :param confidence:
        :param min_c:
        :param max_c:
        """

        super(CW, self).__init__(target)
        self._confidence = confidence
        self._min_c = min_c
        self._max_c = max_c
        self._lr = lr
        self._initial_c = initial_c
        self._max_iter = max_iter
        self._max_binary_step = max_binary_step
        self.n_restart = n_restart
        self.l2_history = []
        self.noprint = noprint

    def save(self, dir_name):
        l2_history = np.asarray(self.l2_history)
        np.save(os.path.join(dir_name, 'cw.npy'), l2_history)

    def compare(self, z, t):
        """
        :param z: logit
        :param t: label to compare
        :return:
        """

        z2 = z.copy()
        z2[list(range(z2.shape[0])), t] -= self._confidence
        return chainer.cuda.cupy.argmax(z2, 1) == t

    def l2_loss(self, image, original_image):
        """
        :param image: current adversarial image
        :param original_image: original image
        :return:
        """

        return chainer.functions.sum((image - original_image) ** 2,
                                     axis=tuple(range(1, image.ndim)))

    def attack_loss(self, image, target, label):
        """
        max(Z(X)_{real} - max{Z(X)_i:i\neq real}, -\kappa)
        :param image: current adversarial image
        :param target: target label
        :return: attack loss
        :rtype: chainer.Variable
        """

        with chainer.force_backprop_mode():
            Z = self.predict(image, label)
        xp = chainer.cuda.get_array_module(Z)
        tmp = xp.ones_like(Z, dtype=xp.float32) * 1e20
        tmp[list(range(tmp.shape[0])), target] = -1e20
        other = chainer.functions.minimum(Z, tmp)
        other = chainer.functions.max(other, 1)
        real = Z[list(range(tmp.shape[0])), target]
        Z_diff = real - other
        return chainer.functions.maximum(Z_diff + self._confidence,
                                         xp.zeros_like(Z_diff,
                                                       dtype=xp.float32)), Z.data

    def loss(self, image, target, original_image, c, label):
        """
        total loss
        note that image must not be a reference to the original image
        :param image: current adversarial iamge
        :param target: target label
        :param original_image: original image
        :param c: control parameter
        :return:
        """

        l2_dist = self.l2_loss(image, original_image)
        attack_loss, logit = self.attack_loss(image, target, label)
        loss = l2_dist + c * attack_loss
        return l2_dist.data, loss.data, logit, chainer.functions.mean(loss)

    def craft(self, image, label):
        """
        image is assumed to be in range [0, 1]
        :param image:
        :param label:
        :return:
        """

        adversarial_image_org = (image - 0.5) * 2. * 0.999999
        xp = chainer.cuda.get_array_module(adversarial_image_org)
        cost = xp.ones(image.shape[0], dtype=xp.float32) * self._initial_c
        upper_bound = xp.ones(label.shape) * self._max_c
        lower_bound = xp.ones(label.shape) * self._min_c
        o_best_l2 = xp.ones(label.shape) * 1e10
        o_best_logit = xp.ones(label.shape) * -1
        o_best_attack = image.copy()
        with chainer.force_backprop_mode():
            for i in range(self._max_binary_step):
                msg_base = '\riter: ' + str(i) + ' '
                best_logit = xp.ones(label.shape) * -1
                best_l2 = xp.ones(label.shape) * 1e10
                for r in range(self.n_restart):
                    if r > 0:
                        start_img = adversarial_image_org + xp.random.normal(
                            scale=1e-2 / float(
                                np.sqrt(np.prod(image.shape[1:]))),
                            size=image.shape).astype(xp.float32)
                    else:
                        start_img = adversarial_image_org
                    adversarial_image = chainer.Parameter(xp.arctanh(start_img))
                    opt = chainer.optimizers.Adam(alpha=self._lr)
                    adversarial_image.update_rule = opt.create_update_rule()
                    for j in range(self._max_iter):
                        # get scores
                        feed_image = (chainer.functions.tanh(
                            adversarial_image) * 0.5 + 0.5)
                        l2_dist, loss_data, logit, loss = self.loss(feed_image,
                                                                    label,
                                                                    image, cost,
                                                                    label)
                        cmp = self.compare(logit, label)
                        cmpl2 = best_l2 > l2_dist
                        change = cmpl2 & ~cmp
                        if change.any():
                            best_l2[change] = l2_dist[change]
                            best_logit[change] = 1
                        o_cmpl2 = o_best_l2 > l2_dist
                        change = o_cmpl2 & ~cmp
                        if change.any():
                            o_best_l2[change] = l2_dist[change]
                            o_best_attack[change] = (xp.tanh(
                                adversarial_image.data[change]) * 0.5 + 0.5)
                            o_best_logit[change] = 1
                        self.cleargrads()
                        adversarial_image.grad = None
                        loss.backward()
                        adversarial_image.update()
                # binary search for cost
                success = o_best_logit == 1
                if not self.noprint:
                    sys.stdout.write(msg_base + 'success: ' + str(success.sum()))
                    sys.stdout.flush()
                success = best_logit == 1
                upper_bound[success] = xp.minimum(upper_bound[success],
                                                  cost[success])
                lower_bound[~success] = xp.maximum(lower_bound[~success],
                                                   cost[~success])
                do_bin_search = upper_bound < (self._max_c - 1)
                cost[do_bin_search] = (upper_bound + lower_bound)[
                                          do_bin_search] * .5
                cost[~do_bin_search & ~success] *= 10
            if not self.noprint:
                sys.stdout.write('\n')
            self.l2_history.extend(list(np.sqrt(o_best_l2.get())))
            return o_best_attack
