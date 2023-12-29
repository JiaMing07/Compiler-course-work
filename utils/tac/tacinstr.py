from enum import Enum, auto, unique
from typing import Any, Optional, Union, List

from utils.label.label import Label
from utils.tac.nativeinstr import NativeInstr
from utils.tac.reg import Reg
from utils.tac.tacop import InstrKind

from .tacop import *
from .tacvisitor import TACVisitor
from .temp import Temp



class TACInstr:
    def __init__(
        self,
        kind: InstrKind,
        dsts: list[Temp],
        srcs: list[Temp],
        label: Optional[Label],
    ) -> None:
        self.kind = kind
        self.dsts = dsts.copy()
        self.srcs = srcs.copy()
        self.label = label

    def getRead(self) -> list[int]:
        # print([i.index for i in self.dsts], self.srcs, self.kind)
        return [src.index for src in self.srcs]

    def getWritten(self) -> list[int]:
        return [dst.index for dst in self.dsts]

    def isLabel(self) -> bool:
        return self.kind is InstrKind.LABEL

    def isSequential(self) -> bool:
        return self.kind == InstrKind.SEQ

    def isReturn(self) -> bool:
        return self.kind == InstrKind.RET

    def toNative(self, dstRegs: list[Reg], srcRegs: list[Reg]) -> NativeInstr:
        oldDsts = dstRegs
        oldSrcs = srcRegs
        self.dsts = dstRegs
        self.srcs = srcRegs
        instrString = self.__str__()
        newInstr = NativeInstr(self.kind, dstRegs, srcRegs, self.label, instrString)
        self.dsts = oldDsts
        self.srcs = oldSrcs
        return newInstr

    def accept(self, v: TACVisitor) -> None:
        pass


# Assignment instruction.
class Assign(TACInstr):
    def __init__(self, dst: Temp, src: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [src], None)
        self.dst = dst
        self.src = src

    def __str__(self) -> str:
        return "%s = %s" % (self.dst, self.src)

    def accept(self, v: TACVisitor) -> None:
        v.visitAssign(self)


# Loading an immediate 32-bit constant.
class LoadImm4(TACInstr):
    def __init__(self, dst: Temp, value: int) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.dst = dst
        self.value = value

    def __str__(self) -> str:
        return "%s = %d" % (self.dst, self.value)

    def accept(self, v: TACVisitor) -> None:
        v.visitLoadImm4(self)


# Unary operations.
class Unary(TACInstr):
    def __init__(self, op: TacUnaryOp, dst: Temp, operand: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [operand], None)
        self.op = op
        self.dst = dst
        self.operand = operand

    def __str__(self) -> str:
        return "%s = %s %s" % (
            self.dst,
            ("-" if (self.op == TacUnaryOp.NEG) else "!"),
            self.operand,
        )

    def accept(self, v: TACVisitor) -> None:
        v.visitUnary(self)


# Binary Operations.
class Binary(TACInstr):
    def __init__(self, op: TacBinaryOp, dst: Temp, lhs: Temp, rhs: Temp) -> None:
        super().__init__(InstrKind.SEQ, [dst], [lhs, rhs], None)
        self.op = op
        self.dst = dst
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self) -> str:
        opStr = {
            TacBinaryOp.ADD: "+",
            TacBinaryOp.SUB: "-",
            TacBinaryOp.MUL: "*",
            TacBinaryOp.DIV: "/",
            TacBinaryOp.MOD: "%",
            TacBinaryOp.EQU: "==",
            TacBinaryOp.NEQ: "!=",
            TacBinaryOp.SLT: "<",
            TacBinaryOp.LEQ: "<=",
            TacBinaryOp.SGT: ">",
            TacBinaryOp.GEQ: ">=",
            TacBinaryOp.AND: "&&",
            TacBinaryOp.OR: "||",
        }[self.op]
        return "%s = (%s %s %s)" % (self.dst, self.lhs, opStr, self.rhs)

    def accept(self, v: TACVisitor) -> None:
        v.visitBinary(self)


# Branching instruction.
class Branch(TACInstr):
    def __init__(self, target: Label) -> None:
        super().__init__(InstrKind.JMP, [], [], target)
        self.target = target

    def __str__(self) -> str:
        return "branch %s" % str(self.target)

    def accept(self, v: TACVisitor) -> None:
        v.visitBranch(self)


# Branching with conditions.
class CondBranch(TACInstr):
    def __init__(self, op: CondBranchOp, cond: Temp, target: Label) -> None:
        super().__init__(InstrKind.COND_JMP, [], [cond], target)
        self.op = op
        self.cond = cond
        self.target = target

    def __str__(self) -> str:
        return "if (%s %s) branch %s" % (
            self.cond,
            "== 0" if self.op == CondBranchOp.BEQ else "!= 0",
            str(self.target),
        )

    def accept(self, v: TACVisitor) -> None:
        v.visitCondBranch(self)


# Return instruction.
class Return(TACInstr):
    def __init__(self, value: Optional[Temp], ident = None) -> None:
        if value is None:
            super().__init__(InstrKind.RET, [], [], None)
        else:
            super().__init__(InstrKind.RET, [], [value], None)
        self.value = value

    def __str__(self) -> str:
        return "return" if (self.value is None) else ("return " + str(self.value))

    def accept(self, v: TACVisitor) -> None:
        v.visitReturn(self)


# Annotation (used for debugging).
class Memo(TACInstr):
    def __init__(self, msg: str) -> None:
        super().__init__(InstrKind.SEQ, [], [], None)
        self.msg = msg

    def __str__(self) -> str:
        return "memo '%s'" % self.msg

    def accept(self, v: TACVisitor) -> None:
        v.visitMemo(self)


# Label (function entry or branching target).
class Mark(TACInstr):
    def __init__(self, label: Label) -> None:
        super().__init__(InstrKind.LABEL, [], [], label)

    def __str__(self) -> str:
        return "%s:" % str(self.label)

    def accept(self, v: TACVisitor) -> None:
        v.visitMark(self)
        
class Param(TACInstr):
    def __init__(self, par: Temp) -> None:
        super().__init__(InstrKind.SEQ, [par], [], None)
        self.par = par

    def __str__(self) -> str:
        return "PARAM %s" % (self.par)

    def accept(self, v: TACVisitor) -> None:
        v.visitParam(self)
        
class CALL(TACInstr):
    def __init__(self, dst: Temp, func: Label, argument_list: List[Temp]) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], func)
        self.dst = dst
        self.func = func
        self.argument_list = argument_list

    def __str__(self) -> str:
        # print("self.func", self.func)
        return f"{self.dst} = CALL {self.func}"

    def accept(self, v: TACVisitor) -> None:
        v.visitCall(self)

class LoadWord(TACInstr):
    def __init__(self, dst: Temp, src: Temp, offset: int, ident = None) -> None:
        super().__init__(InstrKind.SEQ, [dst], [src], None)
        self.offset = offset
        self.ident = ident
        
    def __str__(self) -> str:
        return f"{self.dsts[0]} =  LOAD {self.offset}({self.srcs[0]}) # for {self.ident.value}"
    
    def accept(self, v: TACVisitor) -> None:
        v.visitLoadWord(self)
        
class SaveWord(TACInstr):
    def __init__(self, dst: Temp, src: Temp, offset: int, ident = None) -> None:
        super().__init__(InstrKind.SEQ, [dst], [src], None)
        self.offset = offset
        self.ident = ident
        
    def __str__(self) -> str:
        return f"SAVE {self.dsts[0]}, {self.offset}({self.srcs[0]}) # for {self.ident.value}"
    
    def accept(self, v: TACVisitor) -> None:
        v.visitSaveWord(self)
        
class LoadSymbol(TACInstr):
    def __init__(self, dst: Temp, global_symbol) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.global_symbol = global_symbol
        
    def __str__(self) -> str:
        return f"{self.dsts[0]} =  LOAD_SYMBOL {self.global_symbol.name}"
    
    def accept(self, v: TACVisitor) -> None:
        v.visitLoadSymbol(self)
        
class Alloc(TACInstr):
    def __init__(self, dst: Temp, size: int, ident=None) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.size = size
        self.ident = ident
        
    def __str__(self) -> str:
        return f"{self.dsts[0]} = ALLOC {self.size} # for {self.ident.value}"
    
    def accept(self, v: TACVisitor) -> None:
        return v.visitAlloc(self)
    
class Phi(TACInstr):
    """
    Phi 指令
    """
    def __init__(self, dst: Temp, ident) -> None:
        super().__init__(InstrKind.SEQ, [dst], [], None)
        self.srcs = []
        self.labels = []
        self.ident = ident
    
    def __str__(self) -> str:
        str_phi = ""
        for i in range(len(self.srcs)):
            str_phi += f"({self.srcs[i]}, {self.labels[i]}), "
        return f"{self.dsts[0]} = Phi [{str_phi[:-2]}]  # for {self.ident[1]}"
    
    def add_label(self,label):
        self.labels.append(label)
        self.srcs.append(None)
        
    def add_src(self, src, label):
        idx = self.labels.index(label)
        self.srcs[idx] = src