from backend.dataflow.cfg import CFG
from backend.dataflow.cfgbuilder import CFGBuilder
from backend.dataflow.dominate import Dominate
from backend.dataflow.basicblock import BasicBlock, BlockKind
from backend.dataflow.livenessanalyzer import LivenessAnalyzer
from backend.reg.bruteregalloc import BruteRegAlloc
from backend.riscv.riscvasmemitter import RiscvAsmEmitter
from utils.tac.tacprog import TACProg

class gen_cfg:
    def __init__(self, emitter: RiscvAsmEmitter, regAlloc: BruteRegAlloc) -> None:
        self.emitter = emitter
        self.regAlloc = regAlloc

    def transform(self, prog: TACProg, temps_num, label_id):
        print("transform")
        analyzer = LivenessAnalyzer()

        for idx, func in enumerate(prog.funcs):
            pair = self.emitter.selectInstr(func)
            # print("pair[0]")
            # for ins in pair[0]:
            #     print(ins)
            # print("pair[1]")
            # print(pair[1])
            builder = CFGBuilder()
            cfg: CFG = builder.buildFrom(func.getInstrSeq())
            # for node in cfg.nodes:
            #     print(node.label)
            #     for l in node.locs:
            #         print(l.instr)
            # print(cfg.edges)
            # print(cfg.links)
            dominate = Dominate(cfg, temps_num[idx], pair[1].funcLabel, label_id)
            # analyzer.accept(cfg)
            # self.regAlloc.accept(cfg, pair[1])
            
    def print_locs(block):
        print(block.kind)
        for l in block.locs:
            print(l)

        