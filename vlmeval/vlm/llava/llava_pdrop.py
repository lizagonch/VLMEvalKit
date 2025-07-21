import os.path as osp
import warnings
import logging

from .llava import LLaVA


class LLaVA_PDrop(LLaVA):
    """LLaVA model with PyramidDrop support."""

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(
        self,
        model_path="liuhaotian/llava-v1.5-13b",
        layer_list="[8,16,24]",
        image_token_ratio_list="[0.5,0.25,0.125]",
        **kwargs,
    ):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except Exception as err:
            logging.critical(
                'Please install PyramidDrop from https://github.com/Cooperx521/PyramidDrop before using LLaVA_PDrop'
            )
            raise err

        assert osp.exists(model_path) or len(model_path.split('/')) == 2

        self.system_prompt = (
            'A chat between a curious human and an artificial intelligence assistant. '
            'The assistant gives helpful, detailed, and polite answers to the human\'s questions. '
        )
        self.stop_str = '</s>'

        model_name = get_model_name_from_path(model_path)

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device_map='cpu',
            pdrop_infer=True,
        )

        self.model = self.model.cuda()
        self.conv_mode = 'llava_v1'

        self.model.model.layer_list = eval(layer_list) if isinstance(layer_list, str) else layer_list
        ratios = eval(image_token_ratio_list) if isinstance(image_token_ratio_list, str) else image_token_ratio_list
        self.model.model.image_token_ratio_list = [1.0] + list(ratios)

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config.')
