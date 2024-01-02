from enum import Enum, auto, unique


# Kinds of instructions.
@unique
class InstrKind(Enum):
    # Labels.
    LABEL = auto()
    # Sequential instructions (unary operations, binary operations, etc).
    SEQ = auto()
    # Branching instructions.
    JMP = auto()
    # Branching with conditions.
    COND_JMP = auto()
    # Return instruction.
    RET = auto()
    # PARAMS
    PARAM = auto()
    # Call
    CALL = auto()


# Kinds of unary operations.
@unique
class TacUnaryOp(Enum):
    NEG = auto()
    BitNot = auto()
    LogicNot = auto()

# Kinds of binary operations.
@unique
class TacBinaryOp(Enum):
    ADD = auto()
    SUB = auto()
    
    MUL = auto()
    DIV = auto()
    MOD = auto()
    
    OR = auto()
    AND = auto()
    
    EQU = auto()
    NEQ = auto()
    
    LEQ = auto()
    GEQ = auto()
    SLT = auto()
    SGT = auto()


# Kinds of branching with conditions.
@unique
class CondBranchOp(Enum):
    BEQ = auto()
    BNE = auto()
