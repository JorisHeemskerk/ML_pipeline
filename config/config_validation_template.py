""" Config yaml template

    This template validates the composition of the configuration yaml
    file with `jsonschema`. See the "Config manual" in the README for the
    meaning of every option.

    The template is a catch-all for modules: every module needs a
    `label`, and the parameters of the modules in `MODULE_SCHEMAS` are
    checked as well. A module that is not in there (e.g. a future one) is
    accepted as long as it has an object of parameters with a `label`.
    To get a new module checked too, add it to `MODULE_SCHEMAS`.

    The `general` section accepts unknown keys. The `trainer` section
    does not, so a typo in one of its keys is caught before training.

    Minimal viable yaml file looks like (_'s are placeholders for data):

    ```yaml
    general:
        output_folder: _
        seed: _
    jobs:
        - job:
            name: _
            modules:
                - _ModuleName_:
                    label: _
                    _parameter_: _
            trainer:
                train:
                    data: _
                val:
                    data: _
                    fraction: _     # only if val uses the same data
                test:
                    data: _
                model: _
                epochs: _
                loss:
                    name: _         # or `module: _`
                optimiser:
                    name: _         # or `module: _`
    ```
"""


# Building blocks.
STRING = {'type': 'string'}
BOOLEAN = {'type': 'boolean'}
NUMBER = {'type': 'number'}
OBJECT = {'type': 'object'}
INTEGER = {'type': 'integer'}
POSITIVE_INTEGER = {'type': 'integer', 'minimum': 1}
LABEL = {'type': 'string', 'minLength': 1}

# A torch class by `name` (with `params`), or a module by label.
COMPONENT = {
    'oneOf': [
        {
            'type': 'object',
            'properties': {
                'name': STRING,
                'params': {'type': ['object', 'null']},
            },
            'required': ['name'],
            'additionalProperties': False
        },
        {
            'type': 'object',
            'properties': {'module': LABEL},
            'required': ['module'],
            'additionalProperties': False
        },
    ]
}

# A registered item (metric, visualiser, callback): "name" or
# ["name", {kwargs}].
REGISTERED = {
    'anyOf': [
        STRING,
        {
            'type': 'array',
            'prefixItems': [STRING, {'type': ['object', 'null']}],
            'minItems': 2,
            'maxItems': 2
        },
    ]
}
REGISTERED_LIST = {'type': 'array', 'items': REGISTERED}

# Keyword arguments of a torch DataLoader.
LOADER = {'type': ['object', 'null']}


def _module(properties: dict, required: list[str]=[]) -> dict:
    """
    Schema of the parameters of one module. Every module has a label.
    """
    return {
        'type': 'object',
        'properties': {'label': LABEL, **properties},
        'required': ['label', *required],
        'additionalProperties': False
    }


# Parameters of the modules that are known. Other modules are accepted
# as long as they have a `label` (see MODULE below).
MODULE_SCHEMAS = {
    'Base': _module({}),
    'MNISTData': _module(
        {
            'root': STRING,
            'train': BOOLEAN,
            'normalise': BOOLEAN,
            'download': BOOLEAN,
        },
        required=['root', 'train', 'normalise']
    ),
    'CIFAR10Data': _module(
        {
            'root': STRING,
            'train': BOOLEAN,
            'download': BOOLEAN,
        },
        required=['root', 'train']
    ),
    'NeuralNetwork': _module(
        {
            'network_shape': {
                'type': 'array',
                'minItems': 1,
                # One layer: {LayerName: {kwargs}}, kwargs may be empty.
                'items': {
                    'type': 'object',
                    'minProperties': 1,
                    'maxProperties': 1,
                    'additionalProperties': {'type': ['object', 'null']}
                },
            },
        },
        required=['network_shape']
    ),
    'ResNet': _module({
        'architecture': {
            'enum': [
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
            ]
        },
        'pretrained': BOOLEAN,
        'frozen': BOOLEAN,
        'input_size': {'type': ['integer', 'null'], 'minimum': 1},
    }),
    'SparseAutoencoder': _module(
        {
            'input_dim': POSITIVE_INTEGER,
            'hidden_dim': POSITIVE_INTEGER,
        },
        required=['input_dim', 'hidden_dim']
    ),
    'SparseAutoencoderLoss': _module(
        {'l1_coeff': {'type': 'number', 'minimum': 0}},
        required=['l1_coeff']
    ),
    'Chain': _module(
        {
            'stages': {
                'type': 'array',
                'items': LABEL,
                'minItems': 1
            },
        },
        required=['stages']
    ),
}

# One entry of `modules`: {ModuleName: {parameters}}. Known modules get
# their parameters checked, unknown (future) modules only need a label.
MODULE = {
    'type': 'object',
    'minProperties': 1,
    'maxProperties': 1,
    'properties': MODULE_SCHEMAS,
    'additionalProperties': _module({}) | {'additionalProperties': True},
}

TRAINER = {
    'type': 'object',
    'properties': {
        'mode': {'enum': ['holdout', 'kfold']},
        'loader_defaults': LOADER,
        'train': {
            'type': 'object',
            'properties': {'data': LABEL, 'loader': LOADER},
            'required': ['data'],
            'additionalProperties': False
        },
        'val': {
            'type': 'object',
            'properties': {
                'data': LABEL,
                'loader': LOADER,
                'fraction': {
                    'type': 'number',
                    'exclusiveMinimum': 0,
                    'exclusiveMaximum': 1
                },
                'seed': INTEGER,
            },
            'required': ['data'],
            'additionalProperties': False
        },
        'test': {
            'type': 'object',
            'properties': {
                'data': LABEL,
                'loader': LOADER,
                'visualisers': REGISTERED_LIST,
            },
            'required': ['data'],
            'additionalProperties': False
        },
        'model': LABEL,
        'epochs': POSITIVE_INTEGER,
        'loss': COMPONENT,
        'optimiser': COMPONENT,
        'scheduler': COMPONENT,
        'metrics': REGISTERED_LIST,
        'monitor': {
            'type': 'object',
            'properties': {
                'value': STRING,
                'goal': {'enum': ['min', 'max']},
            },
            'additionalProperties': False
        },
        'early_stopping': {
            'type': ['object', 'null'],
            'properties': {
                'patience': POSITIVE_INTEGER,
                'min_delta': {'type': 'number', 'minimum': 0},
            },
            'additionalProperties': True
        },
        'checkpoint': {
            'type': ['object', 'null'],
            'properties': {
                'every': POSITIVE_INTEGER,
                'save_best': BOOLEAN,
                'plot_curves': BOOLEAN,
                'resume': BOOLEAN,
            },
            'additionalProperties': False
        },
        'callbacks': REGISTERED_LIST,
        'mixed_precision': BOOLEAN,
        'export': {
            'type': ['object', 'null'],
            'properties': {'predictions': BOOLEAN},
            'additionalProperties': False
        },
        'kfold': {
            'type': 'object',
            'properties': {
                'k': {'type': 'integer', 'minimum': 2},
                'stratisfied': BOOLEAN,
            },
            'required': ['k'],
            'additionalProperties': True
        },
    },
    'required': ['train', 'test', 'model', 'epochs', 'loss', 'optimiser'],
    # holdout (the default) needs `val`, kfold needs `kfold`.
    'if': {
        'properties': {'mode': {'const': 'kfold'}},
        'required': ['mode']
    },
    'then': {'required': ['kfold']},
    'else': {'required': ['val']},
    'additionalProperties': False
}

CONFIG_TEMPLATE = {
    'type': 'object',
    'properties': {
        'general': {
            'type': 'object',
            'properties': {
                'output_folder': STRING,
                'device': STRING,
                'seed': INTEGER,
            },
            'required': ['output_folder', 'seed'],
            'additionalProperties': True
        },
        'jobs': {
            'type': 'array',
            'minItems': 1,
            'items': {
                'type': 'object',
                'properties': {
                    'job': {
                        'type': 'object',
                        'properties': {
                            'name': STRING,
                            'modules': {
                                'type': 'array',
                                'items': MODULE,
                                'minItems': 1
                            },
                            'trainer': TRAINER,
                        },
                        'required': ['name', 'modules', 'trainer'],
                        'additionalProperties': False
                    },
                },
                'required': ['job'],
                'additionalProperties': False
            }
        },
    },
    'required': ['general', 'jobs'],
    'additionalProperties': False
}
