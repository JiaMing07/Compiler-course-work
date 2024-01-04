from backend.dataflow.cfg import CFG
from backend.dataflow.cfgbuilder import CFGBuilder
from backend.dataflow.dominate import Dominate
from backend.dataflow.basicblock import BasicBlock, BlockKind
from backend.dataflow.livenessanalyzer import LivenessAnalyzer
from backend.reg.bruteregalloc import BruteRegAlloc
from backend.riscv.riscvasmemitter import RiscvAsmEmitter
from utils.tac.tacprog import TACProg

class gen_cfg:
    """
    完成 SSA 构造算法
    首先使用框架中的 CFGBuilder 构建一个基本的 CFG，
    然后以该 CFG 作为基础构造一个 Dominate 类的对象，
    在 Dominate 类中完成支配树的构建，支配边界的计算，完成 SSA 构造算法
    """
    def __init__(self, emitter: RiscvAsmEmitter, regAlloc: BruteRegAlloc) -> None:
        self.emitter = emitter
        self.regAlloc = regAlloc

    def transform(self, prog: TACProg, temps_num, label_id):

        for idx, func in enumerate(prog.funcs):
            pair = self.emitter.selectInstr(func)
            builder = CFGBuilder()
            cfg: CFG = builder.buildFrom(func.getInstrSeq())
            dominate = Dominate(cfg, temps_num[idx], pair[1].funcLabel, label_id)
            
    def print_locs(block):
        print(block.kind)
        for l in block.locs:
            print(l)

        