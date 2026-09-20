from core.registry import Registry

substrates = Registry()

from .reaction_diffusion import ReactionDiffusion
from .boids import Boids
from .lenia import Lenia
from .nca import NCA
from .controlled_cells import ControlledCells

substrates.register("reaction_diffusion")(ReactionDiffusion)
substrates.register("boids")(Boids)
substrates.register("lenia")(Lenia)
substrates.register("nca")(NCA)
substrates.register("controlled_cells")(ControlledCells)
