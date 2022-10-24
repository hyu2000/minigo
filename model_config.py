import os
from typing import Tuple

import myconf
from evaluate import FLAGS
from katago.analysis_engine import KataModels


class ModelConfig:
    """ to encapsulate model_id + #readouts

    model_config should include #readouts, e.g.
    /path/to/model1_epoch16.h5#200
    """

    def __init__(self, config_str: str):
        self._model_path, self.num_readouts = self._split_model_config(config_str)

    def model_id(self) -> str:
        """ no dir, no .h5 --> model_id#200 """
        if self.is_kata_model():
            return f'%s#{self.num_readouts}' % KataModels.model_id(self._model_path)
        return self._get_model_id(self._model_path, self.num_readouts)

    def model_path(self) -> str:
        """ abs path """
        return self._model_path

    def __str__(self):
        return self.model_id()

    def is_kata_model(self):
        return self._is_kata_model(self._model_path)

    @staticmethod
    def _get_model_id(model_path: str, num_readouts: int) -> str:
        basename = os.path.basename(model_path)
        model_id, _ = os.path.splitext(basename)
        return f'{model_id}#{num_readouts}'

    @staticmethod
    def _is_kata_model(model_path: str):
        return 'kata' in model_path or 'g170' in model_path

    @staticmethod
    def _split_model_config(model_config: str) -> Tuple[str, int]:
        """ figure out abs path, readouts

        model_config should include #readouts, e.g.
        /path/to/model1_epoch16.h5#200
        """
        model_split = model_config.split('#')
        if len(model_split) == 2:
            model_path, num_readouts = model_split[0], int(model_split[1])
        else:
            assert len(model_split) == 1
            model_path, num_readouts = model_config, FLAGS.num_readouts

        if ModelConfig._is_kata_model(model_path):
            assert model_path.startswith('/')
        elif not model_path.startswith('/'):
            default_model_dir = f'{myconf.MODELS_DIR}'
            model_path = f'{default_model_dir}/{model_path}'

            if not model_path.endswith('.mlpackage'):
                model_path = f'{model_path}.mlpackage'
            # backoff to .h5? probably not
            # check file exists

        assert os.path.exists(model_path)
        return model_path, num_readouts
