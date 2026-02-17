import jax
from jax.tree_util import tree_map_with_path, tree_map, GetAttrKey, DictKey, SequenceKey
from dynamax.parameters import ParameterProperties

def _path_to_str(path):
    parts = []
    for key in path:
        if isinstance(key, GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, DictKey):
            parts.append(str(key.key))
        elif isinstance(key, SequenceKey):
            parts.append(str(key.idx))
        else:
            parts.append(str(key))
    return ".".join(parts)

def freeze_all(props):
    """Set trainable=False for all ParameterProperties leaves."""
    def _freeze(prop):
        if isinstance(prop, ParameterProperties):
            return ParameterProperties(trainable=False, constrainer=prop.constrainer)
        return prop
    return tree_map(_freeze, props, is_leaf=lambda x: isinstance(x, ParameterProperties))

def set_trainable_by_paths(props, trainable_paths: list[str], trainable: bool = True):
    """Set trainable flag for specific paths."""
    trainable_set = set(trainable_paths)

    def _update_node(path, prop):
        if isinstance(prop, ParameterProperties):
            path_str = _path_to_str(path)
            # Check if exact match
            if path_str in trainable_set:
                return ParameterProperties(trainable=trainable, constrainer=prop.constrainer)
        return prop

    return tree_map_with_path(_update_node, props, is_leaf=lambda x: isinstance(x, ParameterProperties))

def apply_trainable_paths(props, trainable_paths: list[str]):
    """Freeze all, then unfreeze specified paths."""
    props = freeze_all(props)
    props = set_trainable_by_paths(props, trainable_paths, trainable=True)
    return props
