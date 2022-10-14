#!/home/pengzhiyuan05/.conda/envs/pykaldi/bin/python
# -*- coding: utf-8 -*-

# Author: 2019 - 2022 jerrypeng1937@gmail.com

from collections import OrderedDict
from dataclasses import dataclass


# easy argument parser
@dataclass
class ConfigBase(object):
    silent: bool = True

    def __new__(cls, *args, **kwargs):
        instance = object.__new__(cls)
        instance.__odict__ = OrderedDict()
        return instance

    def __setattr__(self, key, value):
        if key != '__odict__':
            self.__odict__[key] = value
        object.__setattr__(self, key, value)

    def print_args(self, print=print):
        """
        Print all configurations
        """
        print("[Configuration]")
        for key, value in self.__odict__.items():
            print('\'{0}\' : {1}'.format(key, value))
        print('')

    def parse_args(self):
        """
        Supports to pass arguments from command line
        """
        import argparse

        def str2bool(v):
            if v.lower() in ('true', 't'):
                return True
            elif v.lower() in ('false', 'f'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')
        parser = argparse.ArgumentParser()
        for key, value in self.__odict__.items():
            if bool == type(value):
                parser.add_argument('--'+key.replace("-", "_"),
                                    default=str(value), type=str2bool)
            else:
                parser.add_argument('--'+key.replace("-", "_"),
                                    default=value, type=type(value))
        _args = parser.parse_args()
        _args = vars(_args)
        # update
        for key in self.__odict__:
            arg = _args[key]
            self.__odict__[key] = arg
            object.__setattr__(self, key, arg)
        self.normalize()
        if not self.silent:
            self.print_args()
        return self

    def normalize(self):
        pass
