# base
from src.methods.base import BaseModel
from src.methods.base_adios import BaseADIOSModel
from src.methods.linear import LinearModel
from src.methods.supervised import SupervisedModel
# simclr
from src.methods.simclr import SimCLR
from src.methods.simclr_adios import SimCLR_ADIOS
from src.methods.simclr_adios_s import SimCLR_ADIOS_S
from src.methods.simclr_gt import SimCLR_GT
from src.methods.simclr_rand_mask import SimCLR_RM
# simsiam
from src.methods.simsiam import SimSiam
from src.methods.simsiam_adios import SimSiam_ADIOS
from src.methods.simsiam_adios_s import SimSiam_ADIOS_S
# byol
from src.methods.byol import BYOL
from src.methods.byol_adios import BYOL_ADIOS
from src.methods.byol_adios_s import BYOL_ADIOS_S


METHODS = {
    # base classes
    "base": BaseModel,
    "base_adios": BaseADIOSModel,
    "linear": LinearModel,
    "supervised": SupervisedModel,
    # SSL baseline
    "byol": BYOL,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    # SSL + ADIOS
    "simclr_adios": SimCLR_ADIOS,
    "simsiam_adios": SimSiam_ADIOS,
    'byol_adios': BYOL_ADIOS,
    # SSL + ADIOS-s models (single mask)
    "simclr_adios_s": SimCLR_ADIOS_S,
    "simsiam_adios_s": SimSiam_ADIOS_S,
    'byol_adios_s': BYOL_ADIOS_S,
    # other masking schemes
    'simclr_gt': SimCLR_GT,
    'simclr_rand_mask': SimCLR_RM,
}
__all__ = [
    "BaseModel",
    "BaseADIOSModel",
    "LinearModel",
    "SupervisedModel",
    "BYOL",
    "SimCLR",
    "SimSiam",
    "SimCLR_ADIOS",
    "SimSiam_ADIOS",
    "BYOL_ADIOS",
    "SimCLR_ADIOS_S",
    "SimSiam_ADIOS_S",
    "BYOL_ADIOS_S",
    "SimCLR_GT",
    "SimCLR_RM"
]

try:
    from src.methods import dali  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali")
