"""
Testing named keyword args and dicts
"""
import logging

log = logging.getLogger(__name__)


class SomeClass(object):
    """docstring for SomeClass"""

    def __init__(self):
        super(SomeClass, self).__init__()

    def some_function(self, input):

        if getattr(self, 'some_function_compiled', None) is None:
            log.info('Initializing compiled function')
            self.some_function_compiled = lambda x: 2 * x

        return self.some_function_compiled(input)


def f_kwargs(arg0, **kwargs):

    args = {'arg1': 100, 'arg2': 200, 'arg3': 300}
    args.update(kwargs)

    log.info('arg0: %r', arg0)
    for k in args.keys():
        log.info('Kwarg[%s] = %s', k, args[k])


def main():
    logging.basicConfig(level=logging.DEBUG)
    log.info('Woohoo')

    f_kwargs(arg1=10, arg2='asd')
    f_kwargs(**{'arg1': 10, 'arg2': 3.5})

    obj = SomeClass()
    log.info(obj.some_function(2))
    log.info(obj.some_function(3))
    log.info(obj.some_function(4))


if __name__ == '__main__':
    main()
