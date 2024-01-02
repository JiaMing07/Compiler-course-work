from typing import Sequence, Tuple

from backend.asmemitter import AsmEmitter
from utils.error import IllegalArgumentException
from utils.label.label import Label, LabelKind
from utils.riscv import Riscv, RvBinaryOp, RvUnaryOp
from utils.tac.reg import Reg
from utils.tac.tacfunc import TACFunc
from utils.tac.tacinstr import *
from utils.tac.tacinstr import LoadSymbol
from utils.tac.tacvisitor import TACVisitor

from ..subroutineemitter import SubroutineEmitter
from ..subroutineinfo import SubroutineInfo

from frontend.scope.globalscope import GlobalScope
from frontend.symbol.varsymbol import VarSymbol


from math import prod

"""
RiscvAsmEmitter: an AsmEmitter for RiscV
"""


class RiscvAsmEmitter(AsmEmitter):
    def __init__(
        self,
        allocatableRegs: list[Reg],
        callerSaveRegs: list[Reg],
    ) -> None:
        super().__init__(allocatableRegs, callerSaveRegs)
        self.ctx = GlobalScope

    
        # the start of the asm code
        # int step10, you need to add the declaration of global var here
        for name, symbol in GlobalScope.symbols.items():
            if isinstance(symbol, VarSymbol):
                if symbol.isInit:
                    self.printer.println(".data")
                else:
                    self.printer.println(".bss")
                if not symbol.is_array:
                    self.printer.println(f".globl {name}")
                    self.printer.println(f"{name}:")
                    self.printer.println(f"    .word {symbol.initValue[0]}")
                else:
                    self.printer.println(f".globl {name}")
                    self.printer.println(f"{name}:")
                    for val in symbol.initValue:
                        self.printer.println(f"    .word {val}")
                    self.printer.println(f"    .zero {4 * (prod(symbol.dims) - len(symbol.initValue))}")
        
        self.printer.println(".text")
        self.printer.println(".global main")
        self.printer.println("")
            

    # transform tac instrs to RiscV instrs
    # collect some info which is saved in SubroutineInfo for SubroutineEmitter
    def selectInstr(self, func: TACFunc) -> tuple[list[str], SubroutineInfo]:

        selector: RiscvAsmEmitter.RiscvInstrSelector = (
            RiscvAsmEmitter.RiscvInstrSelector(func.entry)
        )
        for i in range(len(func.parameters)):
            temp = func.parameters[i]
            selector.seq.append(Riscv.LoadArgStack(
                temp, Riscv.FP, -(i + 4) * 4
            ))
        
        for instr in func.getInstrSeq():
            instr.accept(selector)

        info = SubroutineInfo(func.entry)
        # print("info", info)

        return (selector.seq, info)

    # use info to construct a RiscvSubroutineEmitter
    def emitSubroutine(self, info: SubroutineInfo):
        return RiscvSubroutineEmitter(self, info)

    # return all the string stored in asmcodeprinter
    def emitEnd(self):
        return self.printer.close()

    class RiscvInstrSelector(TACVisitor):
        def __init__(self, entry: Label) -> None:
            self.entry = entry
            self.seq = []
            self.arg_num = 0

        def visitOther(self, instr: TACInstr) -> None:
            raise NotImplementedError("RiscvInstrSelector visit{} not implemented".format(type(instr).__name__))

        # in step11, you need to think about how to deal with globalTemp in almost all the visit functions. 
        def visitReturn(self, instr: Return) -> None:
            if instr.value is not None:
                self.seq.append(Riscv.Move(Riscv.A0, instr.value))
            else:
                self.seq.append(Riscv.LoadImm(Riscv.A0, 0))
            self.seq.append(Riscv.JumpToEpilogue(self.entry))

        def visitMark(self, instr: Mark) -> None:
            self.seq.append(Riscv.RiscvLabel(instr.label))

        def visitLoadImm4(self, instr: LoadImm4) -> None:
            self.seq.append(Riscv.LoadImm(instr.dst, instr.value))

        def visitUnary(self, instr: Unary) -> None:
            op = {
                TacUnaryOp.NEG: RvUnaryOp.NEG,
                TacUnaryOp.BitNot: RvUnaryOp.NOT,
                TacUnaryOp.LogicNot: RvUnaryOp.SEQZ,
                # You can add unary operations here.
            }[instr.op]
            self.seq.append(Riscv.Unary(op, instr.dst, instr.operand))

        def visitBinary(self, instr: Binary) -> None:
            """
            For different tac operation, you should translate it to different RiscV code
            A tac operation may need more than one RiscV instruction
            """
            if instr.op == TacBinaryOp.OR:
                self.seq.append(Riscv.Binary(RvBinaryOp.OR, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.AND:
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.lhs))
                self.seq.append(Riscv.Binary(RvBinaryOp.SUB, instr.dst, Riscv.ZERO, instr.dst))
                self.seq.append(Riscv.Binary(RvBinaryOp.AND, instr.dst, instr.dst, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.EQU:
                self.seq.append(Riscv.Binary(RvBinaryOp.SUB, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.NEQ:
                self.seq.append(Riscv.Binary(RvBinaryOp.SUB, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.LEQ:
                self.seq.append(Riscv.Binary(RvBinaryOp.SGT, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.GEQ:
                self.seq.append(Riscv.Binary(RvBinaryOp.SLT, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            else:
                op = {
                    TacBinaryOp.ADD: RvBinaryOp.ADD,
                    TacBinaryOp.SUB: RvBinaryOp.SUB,
                    
                    TacBinaryOp.MUL: RvBinaryOp.MUL,
                    TacBinaryOp.DIV: RvBinaryOp.DIV,
                    TacBinaryOp.MOD: RvBinaryOp.REM,
                    
                    TacBinaryOp.SGT: RvBinaryOp.SGT,
                    TacBinaryOp.SLT: RvBinaryOp.SLT,
                    # You can add binary operations here.
                }[instr.op]
                self.seq.append(Riscv.Binary(op, instr.dst, instr.lhs, instr.rhs))

        def visitCondBranch(self, instr: CondBranch) -> None:
            self.seq.append(Riscv.Branch(instr.cond, instr.label))
        
        def visitBranch(self, instr: Branch) -> None:
            self.seq.append(Riscv.Jump(instr.target))
            
        def visitAssign(self, instr: Assign) ->None:
            # print(Riscv.Move(instr.dst, instr.src))
            self.seq.append(Riscv.Move(instr.dst, instr.src))
            
        def visitParam(self, instr: Param) -> None:
            self.seq.append(Riscv.Param(instr.par))
            
        def visitCall(self, instr: CALL) -> None:
            self.seq.append(Riscv.Call(instr.dst, instr.func, instr.argument_list))
            
        def visitLoadWord(self, instr: LoadWord) -> None:
            self.seq.append(Riscv.LoadWord(instr.dsts[0], instr.srcs[0], instr.offset))
            
        def visitSaveWord(self, instr: SaveWord) -> None:
            self.seq.append(Riscv.StoreWord(instr.dsts[0], instr.srcs[0], instr.offset))
            
        def visitLoadSymbol(self, instr: LoadSymbol) -> None:
            self.seq.append(Riscv.LoadSymbol(instr.dsts[0], instr.global_symbol))
        
        def visitAlloc(self, instr: Alloc) -> None:
            self.seq.append(Riscv.Alloc(instr.dsts[0], instr.size))
            
        # in step9, you need to think about how to pass the parameters and how to store and restore callerSave regs
        # in step11, you need to think about how to store the array 
        
"""
RiscvAsmEmitter: an SubroutineEmitter for RiscV
"""

class RiscvSubroutineEmitter(SubroutineEmitter):
    def __init__(self, emitter: RiscvAsmEmitter, info: SubroutineInfo) -> None:
        super().__init__(emitter, info)
        
        # + 4 is for the RA reg 
        self.nextLocalOffset = 4 * len(Riscv.CalleeSaved) + 4
        
        # the buf which stored all the NativeInstrs in this function
        self.buf: list[NativeInstr] = []

        # from temp to int
        # record where a temp is stored in the stack
        self.offsets = {}

        self.printer.printLabel(info.funcLabel)

        # in step9, step11 you can compute the offset of local array and parameters here

    def emitComment(self, comment: str) -> None:
        # you can add some log here to help you debug
        # print(comment)
        pass
    
    # store some temp to stack
    # usually happen when reaching the end of a basicblock
    # in step9, you need to think about the fuction parameters here
    def emitStoreToStack(self, src: Reg) -> None:
        if src.temp.index not in self.offsets:
            self.offsets[src.temp.index] = self.nextLocalOffset - 12
            self.nextLocalOffset += 4
        self.buf.append(
            Riscv.NativeStoreWord(src, Riscv.SP, self.offsets[src.temp.index])
        )

    # load some temp from stack
    # usually happen when using a temp which is stored to stack before
    # in step9, you need to think about the fuction parameters here
    def emitLoadFromStack(self, dst: Reg, src: Temp):
        # print(dst, src, self.offsets)
        if src.index not in self.offsets:
            raise IllegalArgumentException()
        else:
            self.buf.append(
                Riscv.NativeLoadWord(dst, Riscv.SP, self.offsets[src.index])
            )
    
    def emitAlloc(self, dst: Reg, size: int) -> None:
        self.buf.append(
            Riscv.AllocOnStack(dst, self.nextLocalOffset-12)
        )
        self.nextLocalOffset += size

    # add a NativeInstr to buf
    # when calling the fuction emitEnd, all the instr in buf will be transformed to RiscV code
    def emitNative(self, instr: NativeInstr):
        self.buf.append(instr)

    def emitLabel(self, label: Label):
        self.buf.append(Riscv.RiscvLabel(label).toNative([], []))

    
    def emitEnd(self):
        self.printer.printComment("start of prologue")
        self.printer.printInstr(Riscv.NativeStoreWord(Riscv.FP, Riscv.SP, -4))
        self.printer.printInstr(Riscv.NativeStoreWord(Riscv.SP, Riscv.SP, -8))
        self.printer.printInstr(Riscv.NativeStoreWord(Riscv.RA, Riscv.SP, -12))
        self.printer.printInstr(Riscv.NativeLoadWord(Riscv.FP, Riscv.SP, -8))
        self.printer.printInstr(Riscv.SPAdd(-self.nextLocalOffset - 16))
        # in step9, you need to think about how to store RA here
        # you can get some ideas from how to save CalleeSaved regs
        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeStoreWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i)
                )
                
        self.printer.printComment("end of prologue")
        self.printer.println("")

        self.printer.printComment("start of body")

        # in step9, you need to think about how to pass the parameters here
        # you can use the stack or regs

        # using asmcodeprinter to output the RiscV code
        for instr in self.buf:
            self.printer.printInstr(instr)

        self.printer.printComment("end of body")
        self.printer.println("")

        self.printer.printLabel(
            Label(LabelKind.TEMP, self.info.funcLabel.name + Riscv.EPILOGUE_SUFFIX)
        )
        self.printer.printComment("start of epilogue")

        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeLoadWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i)
                )

        self.printer.printInstr(Riscv.SPAdd(self.nextLocalOffset + 16))
        self.printer.printInstr(Riscv.NativeLoadWord(Riscv.RA, Riscv.SP, -12))
        self.printer.printInstr(Riscv.NativeLoadWord(Riscv.SP, Riscv.SP, -8))
        self.printer.printInstr(Riscv.NativeLoadWord(Riscv.FP, Riscv.SP, -4))
        self.printer.printComment("end of epilogue")
        self.printer.println("")

        self.printer.printInstr(Riscv.NativeReturn())
        self.printer.println("")
