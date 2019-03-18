"""config utilities for yml file."""
import os
import sys
import yaml

# singletone
FLAGS = None


class LoaderMeta(type):
    """Constructor for supporting `!include`.
    """
    def __new__(mcs, __name__, __bases__, __dict__):
        """Add include constructer to class."""
        # register the include constructor on the class
        cls = super().__new__(mcs, __name__, __bases__, __dict__)
        cls.add_constructor('!include', cls.construct_include)
        return cls


class Loader(yaml.Loader, metaclass=LoaderMeta):
    """YAML Loader with `!include` constructor.
    """
    def __init__(self, stream):
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)

    def construct_include(self, node):
        """Include file referenced at node."""
        filename = os.path.abspath(
            os.path.join(self._root, self.construct_scalar(node)))
        extension = os.path.splitext(filename)[1].lstrip('.')
        with open(filename, 'r') as f:
            if extension in ('yaml', 'yml'):
                return yaml.load(f, Loader)
            else:
                return ''.join(f.readlines())


class AttrDict(dict):
    """Dict as attribute trick.

    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, list):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value

    def yaml(self):
        """Convert object to yaml dict and return.

        """
        yaml_dict = {}
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                yaml_dict[key] = value.yaml()
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    new_l = []
                    for item in value:
                        new_l.append(item.yaml())
                    yaml_dict[key] = new_l
                else:
                    yaml_dict[key] = value
            else:
                yaml_dict[key] = value
        return yaml_dict

    def __repr__(self):
        """Print all variables.

        """
        ret_str = []
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, AttrDict):
                ret_str.append('{}:'.format(key))
                child_ret_str = value.__repr__().split('\n')
                for item in child_ret_str:
                    ret_str.append('    ' + item)
            elif isinstance(value, list):
                if isinstance(value[0], AttrDict):
                    ret_str.append('{}:'.format(key))
                    for item in value:
                        # treat as AttrDict above
                        child_ret_str = item.__repr__().split('\n')
                        for item in child_ret_str:
                            ret_str.append('    ' + item)
                else:
                    ret_str.append('{}: {}'.format(key, value))
            else:
                ret_str.append('{}: {}'.format(key, value))
        return '\n'.join(ret_str)


class Config(AttrDict):
    """Config with yaml file.

    This class is used to config model hyper-parameters, global constants, and
    other settings with yaml file. All settings in yaml file will be
    automatically logged into file.

    Args:
        filename(str): File name.

    Examples:

        yaml file ``model.yml``::

            NAME: 'neuralgym'
            ALPHA: 1.0
            DATASET: '/mnt/data/imagenet'

        Usage in .py:

        >>> from neuralgym import Config
        >>> config = Config('model.yml')
        >>> print(config.NAME)
            neuralgym
        >>> print(config.ALPHA)
            1.0
        >>> print(config.DATASET)
            /mnt/data/imagenet

    """

    def __init__(self, filename=None, verbose=False):
        assert os.path.exists(filename), 'File {} not exist.'.format(filename)
        try:
            with open(filename, 'r') as f:
                cfg_dict = yaml.load(f, Loader)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        super(Config, self).__init__(cfg_dict)
        if verbose:
            print(' pi.cfg '.center(80, '-'))
            print(self.__repr__())
            print(''.center(80, '-'))


def app():
    """Load app via stdin from subprocess"""
    global FLAGS
    if FLAGS is None:
        job_yaml_file = None
        for arg in sys.argv:
            if arg.startswith('app:'):
                job_yaml_file = arg[4:]
        if job_yaml_file is None:
            job_yaml_file = sys.stdin.readline()
        FLAGS = Config(job_yaml_file)
        return FLAGS
    else:
        return FLAGS


app()
