from core.registry import Registry
foundation_models = Registry()

from .random_embedder import RandomEmbedder  # noqa
from .tiny_vlm_stub import TinyVLMStub       # noqa
from .morphology_judge_stub import MorphologyJudgeStub  # noqa
try:
    from .openclip_adapter import OpenCLIPAdapter  # noqa
except ImportError:
    pass

