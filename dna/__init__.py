"""ALife DNA: versioned genomes (theta + traits), evolution, persistence, expression."""
from dna.genome import Genome, TRAIT_SPEC, GENOME_SCHEMA_VERSION
from dna.evolution import crossover, evolve, mutate
from dna.expression import express_persona
from dna.store import GenomeStore

__all__ = [
    "Genome", "TRAIT_SPEC", "GENOME_SCHEMA_VERSION",
    "mutate", "crossover", "evolve", "express_persona", "GenomeStore",
]
