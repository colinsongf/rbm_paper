"""
Provides means to analyze DBNs
trained in the workflow.
"""

import util
import logging
import workflow as wf


log = logging.getLogger(__name__)


def wf_dbns():
    """
    Returns the DBNs from the workflow for all
    done DbnJobs, mapped to DNB names.
    """

    log.info('Getting workflow DBNs')

    jobs = wf.job_queue()

    def job_dbn(job):
        if not isinstance(job, wf.DbnJob):
            return None
        if not job.is_done():
            return None
        return job.results[0]

    return dict([(str(j), job_dbn(j)) for j in jobs if job_dbn(j) is not None])


def eval_dbns():
    """
    Evaluates DBN performance on the classification task.
    """

    log.info('Will evaluate DBN classification performance')
    _, d_test = wf.get_data()

    for dbn_name, dbn in wf_dbns().items():

        cls_cnt = dbn.class_count
        X_test, y_test = d_test[cls_cnt]
        y_test_pred = dbn.classify(X_test)

        acc = sum(y_test == y_test_pred) / float(len(y_test))
        f1_macro = util.f_macro(y_test, y_test_pred)
        log.info('\nDBN: %r', dbn_name)
        log.info('\tacc: %.2f, f1_macro: %.2f', acc, f1_macro)
        log.info('\tConfusion matrix:\n%r',
                 util.confusion_matrix(y_test, y_test_pred))


def hid_act_per_cls():
    """
    Compiles info about hidden unit activations of the first
    layer of workflow DBNs. Info is presented in log output,
    as well as visualisation of first RBM features.
    Done only with testing data.
    """

    log.info('Hidden layer activation per-class analysis')

    _, d_test = wf.get_data()
    _, _, classes = wf.raw_data

    #   for all the done DBN jobs display 1st layer (0th RBM hidden)
    #   activations for all classes
    for dbn_name, dbn in wf_dbns().items():

        log.info('Working with DBN: %r', dbn_name)

        #   get the first rbm for the dbn
        rbm = dbn.rbms[0]

        #   the testing data
        cls_cnt = dbn.class_count
        X_test, y_test = d_test[cls_cnt]

        log.info('Weight mean %.2f', rbm.W.get_value().mean())
        log.info('Hidden layer mean bias %.2f', rbm.b_hid.get_value().mean())

        #   for all the classes get activations
        for cls_ind in range(len(classes)):

            data = X_test[y_test == cls_ind]
            hid_prb, _ = rbm.hid_given_vis(data)
            log.info('Class: %s, data mean: %.3f, hidden act mean: %.4f',
                     classes[cls_ind], data.mean(), hid_prb.mean())

            #   generate the weight matrix that takes into account act_probs
            hid_prb = hid_prb.mean(axis=0)
            hid_prb = hid_prb / hid_prb.max()
            W = rbm.W.get_value().copy() * hid_prb.reshape((1, rbm.n_hid))
            file_name = wf.DIR_IMG + dbn_name + \
                ' class_' + classes[cls_ind] + '.png'
            util.display_weights(
                W, 32, 24, onscreen=False, image_file_name=file_name)


def main():
    logging.basicConfig(level=logging.INFO)
    hid_act_per_cls()

if __name__ == '__main__':
    main()
