"""Learned dynamics models for the HNN double pendulum replication.

Both models implement the same forward contract: given a canonical state
``(q, p)`` of shape ``(..., 4)``, return its time derivative of the same shape.
This lets the evaluation scripts feed either network into the same integrator
loop without special-casing.
"""

from models.mlp_baseline import MLPBaseline
from models.hnn import HNN

__all__ = ["MLPBaseline", "HNN"]
