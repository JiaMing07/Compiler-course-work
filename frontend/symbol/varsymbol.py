from utils.tac.temp import Temp

from typing import List

from .symbol import *


"""
Variable symbol, representing a variable definition.
"""


class VarSymbol(Symbol):
    def __init__(self, name: str, type: DecafType, isGlobal: bool = False, isInit: bool = False) -> None:
        super().__init__(name, type)
        self.temp: Temp
        self.isGlobal = isGlobal
        self.initValue = []
        self.isInit = isInit
        self.is_array = False
        self.is_element = False
        self.dims = []

    def __str__(self) -> str:
        return "variable %s : %s" % (self.name, str(self.type))

    # To set the initial value of a variable symbol (used for global variable).
    def setInitValue(self, value: List[int]) -> None:
        self.initValue = value
