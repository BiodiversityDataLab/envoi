_REG = {}


def register(name, cls):
    _REG[name] = cls


def get_adapter(name):
    return _REG[name]
