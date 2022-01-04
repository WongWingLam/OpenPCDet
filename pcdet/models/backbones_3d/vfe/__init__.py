from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE, PillarVFE_TANet
from .image_vfe import ImageVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'ImageVFE': ImageVFE,
    'PillarVFE_TANet': PillarVFE_TANet
}
