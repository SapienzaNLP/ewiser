import importlib

import os

from fairseq.criterions import CRITERION_REGISTRY

# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('ewiser.fairseq_ext.criterions.' + module)
