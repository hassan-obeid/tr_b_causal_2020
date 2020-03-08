# -*- coding: utf-8 -*-
import abc
from typing import Union, Tuple
from pandas import DataFrame
from graphviz import Digraph

from causalgraphicalmodels import CausalGraphicalModel, StructuralCausalModel

SIZE_TYPE = Union[Tuple[int], int]


class AbstractCausalModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass) -> bool:
        return (
        hasattr(subclass, 'draw') and callable(subclass.draw) and
        hasattr(subclass, 'sample') and callable(subclass.sample))

    @abc.abstractmethod
    def draw(self) -> Digraph:
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, size: SIZE_TYPE) -> DataFrame:
        """Extract text from the data set"""
        raise NotImplementedError


class CGM_Model(AbstractCausalModel, StructuralCausalModel):
    """
    Now, one's causal graphical models can inherit directly from this class.
    """
    def draw(self) -> Digraph:
        return self.cgm.draw()

    def sample(self, size: SIZE_TYPE) -> DataFrame:
        return super(StructuralCausalModel, self).sample(n_samples=size)
