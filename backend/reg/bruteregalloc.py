import random

from backend.dataflow.basicblock import BasicBlock, BlockKind
from backend.dataflow.cfg import CFG
from backend.dataflow.loc import Loc
from backend.reg.regalloc import RegAlloc
from backend.riscv.riscvasmemitter import RiscvAsmEmitter
from backend.subroutineemitter import SubroutineEmitter
from backend.subroutineinfo import SubroutineInfo
from utils.riscv import Riscv
from utils.tac.reg import Reg
from utils.tac.temp import Temp

"""
BruteRegAlloc: one kind of RegAlloc

bindings: map from temp.index to Reg

we don't need to take care of GlobalTemp here
because we can remove all the GlobalTemp in selectInstr process

1. accept：根据每个函数的 CFG 进行寄存器分配，寄存器分配结束后生成相应汇编代码
2. bind：将一个 Temp 与寄存器绑定
3. unbind：将一个 Temp 与相应寄存器解绑定
4. localAlloc：根据数据流对一个 BasicBlock 内的指令进行寄存器分配
5. allocForLoc：每一条指令进行寄存器分配
6. allocRegFor：根据数据流决定为当前 Temp 分配哪一个寄存器
"""

class BruteRegAlloc(RegAlloc):
    def __init__(self, emitter: RiscvAsmEmitter) -> None:
        super().__init__(emitter)
        self.bindings = {}
        for reg in emitter.allocatableRegs:
            reg.used = False

    def accept(self, graph: CFG, info: SubroutineInfo) -> None:
        subEmitter = self.emitter.emitSubroutine(info)
        for bb in graph.iterator():
            # you need to think more here
            # maybe we don't need to alloc regs for all the basic blocks
            if bb.label is not None:
                subEmitter.emitLabel(bb.label)
            self.localAlloc(bb, subEmitter)
        subEmitter.emitEnd()

    def bind(self, temp: Temp, reg: Reg):
        reg.used = True
        self.bindings[temp.index] = reg
        reg.occupied = True
        reg.temp = temp

    def unbind(self, temp: Temp):
        if temp.index in self.bindings:
            self.bindings[temp.index].occupied = False
            self.bindings.pop(temp.index)

    def localAlloc(self, bb: BasicBlock, subEmitter: SubroutineEmitter):
        self.bindings.clear()
        for reg in self.emitter.allocatableRegs:
            reg.occupied = False

        # in step9, you may need to think about how to store callersave regs here
        for loc in bb.allSeq():
            if isinstance(loc.instr, Riscv.Call):
                # print("call")
                ret = loc.instr.dsts[0]
                # 保存 caller_save 寄存器
                saved_regs = []
                sum = 0
                # print("T1",Riscv.T1.occupied, Riscv.T1.temp.index)
                # print("T0",Riscv.T0.occupied, Riscv.T0.temp.index)
                for reg in Riscv.CallerSaved:
                    if reg.occupied and reg.temp.index in loc.liveIn:
                        # print("reg", reg)
                        subEmitter.emitStoreToStack(reg)
                        saved_regs.append([reg.temp, reg])
                        self.unbind(reg.temp)
                        
                # 传参
                # print("T1",Riscv.T0.occupied, Riscv.T0.temp.index)
                for idx, arg in enumerate(loc.instr.argument_list):
                    reg = self.allocRegFor(arg, True, loc.liveIn, subEmitter)
                    sw = Riscv.NativeStoreWord(reg, Riscv.SP, -4*(idx + 4))
                    # print(sw.instrString)
                    subEmitter.emitNative(sw)
                    self.unbind(arg)
                        
                # 调用
                # print("T1",Riscv.T0.occupied, Riscv.T0.temp.index)
                subEmitter.emitNative(loc.instr.toNative([], []))
                # print([instr.instrString for instr in subEmitter.buf])
                # print(ret, loc.instr.dsts)
                self.bind(ret, Riscv.A0)
                
                # 恢复 caller_saved 寄存器
                # print("T1",Riscv.T0.occupied, Riscv.T0.temp.index)
                for temp_reg in saved_regs:
                    if temp_reg[1].occupied:
                        # print("occ", temp_reg[1], temp_reg[1].temp.index)
                        subEmitter.emitStoreToStack(temp_reg[1])
                        self.unbind(temp_reg[1].temp)
                    subEmitter.emitLoadFromStack(temp_reg[1], temp_reg[0])
                    self.bind(temp_reg[0], temp_reg[1])
                    
                # print([instr.instrString for instr in subEmitter.buf])
             
            else:
                subEmitter.emitComment(str(loc.instr))

                self.allocForLoc(loc, subEmitter)
            # print("loc", loc.instr)
            # print([instr.instrString for instr in subEmitter.buf])   

        for tempindex in bb.liveOut:
            if tempindex in self.bindings:
                subEmitter.emitStoreToStack(self.bindings.get(tempindex))

        if (not bb.isEmpty()) and (bb.kind is not BlockKind.CONTINUOUS):
            self.allocForLoc(bb.locs[len(bb.locs) - 1], subEmitter)

    def allocForLoc(self, loc: Loc, subEmitter: SubroutineEmitter):
        instr = loc.instr
        srcRegs: list[Reg] = []
        dstRegs: list[Reg] = []

        for i in range(len(instr.srcs)):
            temp = instr.srcs[i]
            if isinstance(temp, Reg):
                srcRegs.append(temp)
            else:
                srcRegs.append(self.allocRegFor(temp, True, loc.liveIn, subEmitter))

        for i in range(len(instr.dsts)):
            temp = instr.dsts[i]
            if isinstance(temp, Reg):
                dstRegs.append(temp)
            else:
                dstRegs.append(self.allocRegFor(temp, False, loc.liveIn, subEmitter))

        subEmitter.emitNative(instr.toNative(dstRegs, srcRegs))

    def allocRegFor(
        self, temp: Temp, isRead: bool, live: set[int], subEmitter: SubroutineEmitter
    ):
        if temp.index in self.bindings:
            return self.bindings[temp.index]

        for reg in self.emitter.allocatableRegs:
            if (not reg.occupied) or (not reg.temp.index in live):
                # print("reg", reg.index, reg)
                subEmitter.emitComment(
                    "  allocate {} to {}  (read: {}):".format(
                        str(temp), str(reg), str(isRead)
                    )
                )
                if isRead:
                    subEmitter.emitLoadFromStack(reg, temp)
                if reg.occupied:
                    self.unbind(reg.temp)
                self.bind(temp, reg)
                return reg

        reg = self.emitter.allocatableRegs[
            random.randint(0, len(self.emitter.allocatableRegs) - 1)
        ]
        subEmitter.emitStoreToStack(reg)
        subEmitter.emitComment("  spill {} ({})".format(str(reg), str(reg.temp)))
        self.unbind(reg.temp)
        self.bind(temp, reg)
        subEmitter.emitComment(
            "  allocate {} to {} (read: {})".format(str(temp), str(reg), str(isRead))
        )
        if isRead:
            subEmitter.emitLoadFromStack(reg, temp)
        return reg
