from workflow import DbnPretrainJob, get_data
import logging

log = logging.getLogger(__name__)

#   a gloabal variable that holds the job_queue list
#   returned by the job_queue() function
__job_queue = None


def job_queue():
    """
    Returns a list of jobs (some done, some possibly not).
    """

    #   lazy init of a global variable
    global __job_queue
    if __job_queue is not None:
        return __job_queue

    d_train, d_test = get_data()

    #   jobs, params for RBM training are:
    #   (classes, n_vis, n_hid, epochs, epsilon, pcd, steps, spars, spars_cost)
    n_vis = 32 * 24
    # R = RbmJob
    D = DbnPretrainJob
    __job_queue = (
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.1, 0.5), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.1, 0.1), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.1, 0.005), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.2, 0.5), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.2, 0.1), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, False, 1, 0.2, 0.005), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, True, 2, 0.2, 0.5), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, True, 2, 0.2, 0.1), d_train[1][0]),
        # R((1, n_vis, 144, 50, 0.05, True, 2, 0.2, 0.005), d_train[1][0]),

        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.05, 0.25), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.05, 0.075), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.05, 0.025), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.1, 0.25), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.1, 0.1), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.1, 0.025), d_train[3][0]),

        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.075, 0.25), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.075, 0.075), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.125, 0.25), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, False, 1, 0.125, 0.075), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, True, 2, 0.125, 0.25), d_train[3][0]),
        # R((3, n_vis, 276, 50, 0.05, True, 2, 0.125, 0.075), d_train[3][0]),

        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.10, 0.15), d_train[9][0]),
        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.10, 0.3), d_train[9][0]),
        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.075, 0.15), d_train[9][0]),
        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.075, 0.3), d_train[9][0]),
        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.05, 0.15), d_train[9][0]),
        # R((9, n_vis, 432, 50, 0.05, False, 1, 0.05, 0.3), d_train[9][0]),

        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.10, 0.15), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.10, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 2, 0.10, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.075, 0.15), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.075, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.05, 0.15), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, False, 1, 0.05, 0.3), d_train[9][0]),

        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.10, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.075, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 1, 0.05, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.05, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 4, 0.05, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 15, 0.05, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.035, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.065, 0.3), d_train[9][0]),
        # R((9, n_vis, 588, 50, 0.05, True, 2, 0.0, 0.0), d_train[9][0]),

        #   some DBN training!
        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.05, 0.05],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.05, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.085, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        D((
            9, [n_vis, 588, 500],
            [
                [100, 0.05, True, 1, 0.085, 0.15],
                [50, 0.05, True, 1, 0.0, 0.0]
            ]
        ), *d_train[9]),

        D((
            9, [n_vis, 588, 500],
            [
                [30, 0.05, True, 1, 0.085, 0.15],
                [50, 0.05, True, 1, 0.0, 0.0]
            ]
        ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.085, 0.05],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.1, 0.05],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.1, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.15, 0.05],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.15, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     9, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.0, 0.0],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[9]),

        # D((
        #     7, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.05, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[7]),

        # D((
        #     7, [n_vis, 588, 500],
        #     [
        #         [100, 0.05, True, 1, 0.05, 0.3],
        #         [100, 0.03, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[7]),

        # D((
        #     7, [n_vis, 588, 1000],
        #     [
        #         [100, 0.05, True, 1, 0.05, 0.3],
        #         [50, 0.05, True, 1, 0.0, 0.0]
        #     ]
        # ), *d_train[7])

    )

    return __job_queue


def main():

    logging.basicConfig(level=logging.DEBUG)
    log.info('Workflow main()')

    for job in job_queue():
        log.info('Evaluating job: %s', job)
        if not job.is_done():
            job.perform()


if __name__ == '__main__':
    main()
