"""
A script for analyzing hidden layer
activations in RBMs.
"""

import util
import workflow
import logging

log = logging.getLogger(__name__)


def main():

    logging.basicConfig(level=logging.DEBUG)

    log.info('Hidden layer activation analysis')

    X, y, classes = util.load_trainset()

    #   for all the done DBN jobs display 1st layer (0th RBM hidden)
    #   activations for all classes
    for job in workflow.job_queue():

        if ((not isinstance(job, workflow.DbnJob)) |
                (not job.is_done())):
            continue

        log.info('Working with DBN: %r', job)

        #   get the first rbm for the dbn
        rbm = job.results[0].rbms[0]

        log.info('Weight mean %.2f', rbm.W.get_value().mean())
        log.info('Hidden layer mean bias %.2f', rbm.b_hid.get_value().mean())

        #   for all the classes get activations
        for cls_ind in range(len(classes)):

            data = X[y == cls_ind]
            hid_prb, _ = rbm.hid_given_vis(data)
            log.info('Class: %s, data mean: %.3f, hidden act mean: %.4f',
                     classes[cls_ind], data.mean(), hid_prb.mean())

    #   display the mean for '_BLANK class'

if __name__ == '__main__':
    main()
