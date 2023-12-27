from frontend.ast.node import T, Optional
from frontend.ast.tree import T, ArrayElement, Function, Optional
from frontend.ast import node
from frontend.ast.tree import *
from frontend.ast.visitor import T, Visitor
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from frontend.type.builtin_type import BuiltinType
from utils.label.blocklabel import BlockLabel
from utils.label.funclabel import FuncLabel
from utils.tac import tacop
from utils.tac.temp import Temp
from utils.tac.tacinstr import *
from utils.tac.tacfunc import TACFunc
from utils.tac.tacprog import TACProg
from utils.tac.tacvisitor import TACVisitor


def_dic = {}

"""
The TAC generation phase: translate the abstract syntax tree into three-address code.
"""


class LabelManager:
    """
    A global label manager (just a counter).
    We use this to create unique (block) labels accross functions.
    """

    def __init__(self):
        self.nextTempLabelId = 0

    def freshLabel(self) -> BlockLabel:
        self.nextTempLabelId += 1
        return BlockLabel(str(self.nextTempLabelId))


class TACFuncEmitter(TACVisitor):
    """
    Translates a minidecaf (AST) function into low-level TAC function.
    """

    def __init__(
        self, entry: FuncLabel, numArgs: int, labelManager: LabelManager
    ) -> None:
        self.labelManager = labelManager
        self.func = TACFunc(entry, numArgs)
        self.visitLabel(entry)
        self.nextTempId = 0

        self.continueLabelStack = []
        self.breakLabelStack = []

    # To get a fresh new temporary variable.
    def freshTemp(self) -> Temp:
        temp = Temp(self.nextTempId)
        self.nextTempId += 1
        return temp

    # To get a fresh new label (for jumping and branching, etc).
    def freshLabel(self) -> Label:
        return self.labelManager.freshLabel()

    # To count how many temporary variables have been used.
    def getUsedTemp(self) -> int:
        return self.nextTempId

    # In fact, the following methods can be named 'appendXXX' rather than 'visitXXX'.
    # E.g., by calling 'visitAssignment', you add an assignment instruction at the end of current function.
    def visitAssignment(self, dst: Temp, src: Temp) -> Temp:
        self.func.add(Assign(dst, src))
        return src

    def visitLoad(self, value: Union[int, str]) -> Temp:
        temp = self.freshTemp()
        self.func.add(LoadImm4(temp, value))
        return temp

    def visitUnary(self, op: UnaryOp, operand: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(Unary(op, temp, operand))
        return temp

    def visitUnarySelf(self, op: UnaryOp, operand: Temp) -> None:
        self.func.add(Unary(op, operand, operand))

    def visitBinary(self, op: BinaryOp, lhs: Temp, rhs: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(Binary(op, temp, lhs, rhs))
        return temp

    def visitBinarySelf(self, op: BinaryOp, lhs: Temp, rhs: Temp) -> None:
        self.func.add(Binary(op, lhs, lhs, rhs))

    def visitBranch(self, target: Label) -> None:
        self.func.add(Branch(target))

    def visitCondBranch(self, op: CondBranchOp, cond: Temp, target: Label) -> None:
        self.func.add(CondBranch(op, cond, target))

    def visitReturn(self, value: Optional[Temp], ident = None) -> None:
        self.func.add(Return(value))

    def visitLabel(self, label: Label) -> None:
        self.func.add(Mark(label))
        
    def visitParam(self, par: Temp) -> None:
        self.func.add(Param(par))
        
    def visitCall(self, dst: Temp,func: Label, argument_list:List[Temp]) -> None:
        self.func.add(CALL(dst, func, argument_list))

    def visitMemo(self, content: str) -> None:
        self.func.add(Memo(content))

    def visitRaw(self, instr: TACInstr) -> None:
        self.func.add(instr)
        
    def visitLoadWord(self, dst:Temp, src: Temp, offset:int, ident: Identifier = None) -> None:
        self.func.add(LoadWord(dst, src, offset, ident))
        
    def visitSaveWord(self, dst:Temp, src: Temp, offset:int, ident: Identifier = None) -> None:
        self.func.add(SaveWord(dst, src, offset, ident))
        
    def visitLoadSymbol(self, dst: Temp, global_symbol: VarSymbol) -> None:
        self.func.add(LoadSymbol(dst, global_symbol))
        
    def visitArrayElement(self, dst: Temp, src: Temp, size: int) -> None:
        size_temp = self.freshTemp()
        self.func.add(LoadImm4(size_temp, size))
        self.func.add(Binary(TacBinaryOp.MUL, size_temp, size_temp, src))
        self.func.add(Binary(TacBinaryOp.ADD, dst, dst, size_temp))
        
    def visitAlloc(self, dst: Temp, size: int, ident: Identifier = None):
        self.func.add(Alloc(dst, size, ident))

    def visitEnd(self, param_list: list[Temp]) -> TACFunc:
        if (len(self.func.instrSeq) == 0) or (not self.func.instrSeq[-1].isReturn()):
            self.func.add(Return(None))
        self.func.tempUsed = self.getUsedTemp()
        self.func.set_parameters(param_list)
        return self.func
    
    def visitFilln(self, array: Temp, decl: Declaration) -> None:
        address = array
        size = int(decl.var_t.type.size / 4)
        zero_temp = self.visitLoad(0)
        size_temp = self.visitLoad(size)
        self.visitParam(address)
        self.visitParam(zero_temp)
        self.visitParam(size_temp)
        label = FuncLabel("fill_n")
        ret_temp = self.freshTemp()
        self.func.add(CALL(ret_temp, label, [address, zero_temp, size_temp]))

    # To open a new loop (for break/continue statements)
    def openLoop(self, breakLabel: Label, continueLabel: Label) -> None:
        self.breakLabelStack.append(breakLabel)
        self.continueLabelStack.append(continueLabel)

    # To close the current loop.
    def closeLoop(self) -> None:
        self.breakLabelStack.pop()
        self.continueLabelStack.pop()

    # To get the label for 'break' in the current loop.
    def getBreakLabel(self) -> Label:
        return self.breakLabelStack[-1]

    # To get the label for 'continue' in the current loop.
    def getContinueLabel(self) -> Label:
        return self.continueLabelStack[-1]
    
    # print log
    def print_log(self):
        self.func.printTo()


class TACGen(Visitor[TACFuncEmitter, None]):
    # Entry of this phase
    def transform(self, program: Program) -> TACProg:
        labelManager = LabelManager()
        tacFuncs = []
        
        # for glob_name, glob_var in program.globals().items():
            
        temps_num = []
        for funcName, astFunc in program.functions().items():
            # in step9, you need to use real parameter count
            emitter = TACFuncEmitter(FuncLabel(funcName), len(astFunc.params), labelManager)
            param_list = []
            for param in astFunc.params:
                param.accept(self, emitter)
                temp = param.getattr("symbol").temp
                # print(param.ident.value, temp)
                param_list.append(temp)
            # print(param_list)
            astFunc.body.accept(self, emitter)
            tacFuncs.append(emitter.visitEnd(param_list))
            temps_num.append(emitter.nextTempId)
        return TACProg(tacFuncs), temps_num, emitter.labelManager.nextTempLabelId

    def visitBlock(self, block: Block, mv: TACFuncEmitter) -> None:
        for child in block:
            child.accept(self, mv)

    def visitReturn(self, stmt: Return, mv: TACFuncEmitter) -> None:
        stmt.expr.accept(self, mv)
        val = stmt.expr.getattr("val")
        if isinstance(stmt.expr, Identifier):
            temp = mv.freshTemp()
            mv.visitLoadWord(temp, stmt.expr.getattr('address'), 0, stmt.expr)
            val = temp
        mv.visitReturn(val)

    def visitBreak(self, stmt: Break, mv: TACFuncEmitter) -> None:
        mv.visitBranch(mv.getBreakLabel())
        
    def visitContinue(self, stmt: Continue, mv: TACFuncEmitter) -> None:
        mv.visitBranch(mv.getContinueLabel())

    def visitIdentifier(self, ident: Identifier, mv: TACFuncEmitter) -> None:
        """
        1. Set the 'val' attribute of ident as the temp variable of the 'symbol' attribute of ident.
        """
        sym = ident.getattr('symbol')
        # print(sym, sym.is)
        if sym.isGlobal:
            address = mv.freshTemp()
            mv.visitLoadSymbol(address, sym)
            if not sym.is_array:
                temp = mv.freshTemp()
                mv.visitLoadWord(temp, address, 0)
                sym.temp = temp
                ident.setattr('symbol', sym)
                ident.setattr('val', sym.temp)
            else:
                sym.temp = address
                ident.setattr('symbol', sym)
                ident.setattr('val', address)
        else:
        # print(ident.value,ident.getattr('symbol'))
            # print("ident", ident.value, ident.getattr("address"))
            temp = ident.getattr('symbol').temp
            ident.setattr('address', temp)
        # raise NotImplementedError

    def visitDeclaration(self, decl: Declaration, mv: TACFuncEmitter) -> None:
        """
        1. Get the 'symbol' attribute of decl.
        2. Use mv.freshTemp to get a new temp variable for this symbol.
        3. If the declaration has an initial value, use mv.visitAssignment to set it.
        """
        # print(decl.ident.value)
        symbol = decl.getattr("symbol")
        symbol.temp = mv.freshTemp()
        # print("decl", symbol, decl.ident.value)
        decl.setattr("address", symbol.temp)
        decl.setattr("symbol", symbol)
        decl.ident.setattr("address", symbol.temp)
        # print(decl.ident, decl.ident.getattr("address"))
        if decl.is_array:
            mv.visitAlloc(symbol.temp, symbol.type.size)
            if decl.init_expr is not NULL:
                mv.visitFilln(symbol.temp, decl)
                for i, val in enumerate(decl.init_expr.value):
                    val_temp = mv.visitLoad(int(val))
                    mv.visitSaveWord(val_temp, symbol.temp, i * 4)
        else:    
            mv.visitAlloc(symbol.temp, 4, decl.ident)
            if decl.init_expr is not NULL:
                decl.init_expr.accept(self, mv)
                if isinstance(decl.init_expr, Identifier):
                    temp = mv.freshTemp()
                    mv.visitLoadWord(temp, decl.init_expr.getattr('address'), 0, decl.init_expr)
                    mv.visitSaveWord(temp, decl.getattr("address"), 0, decl.ident)
                else:
                    mv.visitSaveWord(decl.init_expr.getattr('val'), decl.getattr("address"), 0, decl.ident)
                # mv.visitAssignment(decl.getattr("symbol").temp, decl.init_expr.getattr('val'))

        # raise NotImplementedError
    def visitParameter(self, param: Parameter, mv: TACFuncEmitter) -> None:
        """
        1. Get the 'symbol' attribute of decl.
        2. Use mv.freshTemp to get a new temp variable for this symbol.
        3. If the declaration has an initial value, use mv.visitAssignment to set it.
        """
        symbol = param.getattr("symbol")
        symbol.temp = mv.freshTemp()
        param.setattr("symbol", symbol)
        if param.init_expr is not NULL:
            param.init_expr.accept(self, mv)
            mv.visitAssignment(param.getattr("symbol").temp, param.init_expr.getattr('val'))

    def visitAssignment(self, expr: Assignment, mv: TACFuncEmitter) -> None:
        """
        1. Visit the right hand side of expr, and get the temp variable of left hand side.
        2. Use mv.visitAssignment to emit an assignment instruction.
        3. Set the 'val' attribute of expr as the value of assignment instruction.
        """
        expr.rhs.accept(self, mv)
        val = expr.rhs.getattr("val")
        expr.lhs.accept(self, mv)
        lhs_sym = expr.lhs.getattr("symbol")
        lhs_address = expr.lhs.getattr("address")
        # print("lhs_address", lhs_address, type(expr.lhs), expr.lhs.getattr("val"))
        # print(expr.lhs, lhs_sym, type(expr.lhs))
        if lhs_sym.isGlobal:
            address = mv.freshTemp()
            if not lhs_sym.is_array:
                mv.visitLoadSymbol(address, lhs_sym)
                temp = mv.freshTemp()
                mv.visitAssignment(temp, expr.rhs.getattr('val'))
                mv.visitSaveWord(temp, address, 0, expr.lhs)
                lhs_sym.temp = temp
                expr.lhs.setattr('symbol', lhs_sym)
                expr.lhs.setattr('val', lhs_sym.temp)
                expr.setattr('val', val)
            else:
                mv.visitLoadSymbol(address, lhs_sym)
                decaf_type = lhs_sym.type
                for idx in expr.lhs.indexes:
                    idx.accept(self, mv)
                    if isinstance(decaf_type, BuiltinType):
                        size = 4
                    else:
                        size = decaf_type.base.size
                        decaf_type = decaf_type.base
                    mv.visitArrayElement(address, idx.getattr("val"), size)
                mv.visitSaveWord(val, address, 0)
                expr.setattr('val', val)
        else:
            if lhs_sym.is_array:
                address = mv.freshTemp()
                mv.visitAssignment(address, lhs_sym.temp)
                decaf_type = lhs_sym.type
                for idx in expr.lhs.indexes:
                    idx.accept(self, mv)
                    if isinstance(decaf_type, BuiltinType):
                        size = 4
                    else:
                        size = decaf_type.base.size
                        decaf_type = decaf_type.base
                    mv.visitArrayElement(address, idx.getattr("val"), size)
                    
                mv.visitSaveWord(val, address, 0)
                expr.setattr('val', val)
            else:
                temp = expr.lhs.getattr('symbol').temp
                if isinstance(expr.rhs, Identifier):
                    rhs_temp = mv.freshTemp()
                    mv.visitLoadWord(rhs_temp, expr.rhs.getattr("address"), 0, expr.rhs)
                    mv.visitSaveWord(rhs_temp, lhs_address, 0, expr.lhs)
                    expr.setattr('val', expr.rhs.getattr('val'))
                else:
                    mv.visitSaveWord(expr.rhs.getattr('val'), lhs_address, 0, expr.lhs)
                    expr.setattr('val', expr.rhs.getattr('val'))
                # mv.visitAssignment(temp, expr.rhs.getattr('val'))
                # expr.setattr('val', expr.rhs.getattr('val'))
        # raise NotImplementedError

    def visitIf(self, stmt: If, mv: TACFuncEmitter) -> None:
        stmt.cond.accept(self, mv)

        if stmt.otherwise is NULL:
            skipLabel = mv.freshLabel()
            mv.visitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitLabel(skipLabel)
        else:
            skipLabel = mv.freshLabel()
            exitLabel = mv.freshLabel()
            mv.visitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitBranch(exitLabel)
            mv.visitLabel(skipLabel)
            stmt.otherwise.accept(self, mv)
            mv.visitLabel(exitLabel)

    def visitWhile(self, stmt: While, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)

        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)

        stmt.body.accept(self, mv)
        mv.visitLabel(loopLabel)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()
        
    def visitDoWhile(self, stmt: DoWhile, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)

        mv.visitLabel(beginLabel)
        stmt.body.accept(self, mv)
        # mv.visitLabel(loopLabel)
        
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)
        mv.visitLabel(loopLabel)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)

        mv.closeLoop()
        
    def visitFor(self, stmt: For, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        
        stmt.init.accept(self, mv)
        
        mv.openLoop(breakLabel, loopLabel)
        mv.visitLabel(beginLabel)
        
        stmt.cond.accept(self, mv)
        if stmt.cond.getattr("val") is not None:
            mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)
        stmt.body.accept(self, mv)
        
        mv.visitLabel(loopLabel)
        stmt.incr.accept(self, mv)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)

        mv.closeLoop()

    def visitUnary(self, expr: Unary, mv: TACFuncEmitter) -> None:
        expr.operand.accept(self, mv)

        op = {
            node.UnaryOp.Neg: tacop.TacUnaryOp.NEG,
            # You can add unary operations here.
            node.UnaryOp.BitNot: tacop.TacUnaryOp.BitNot,
            node.UnaryOp.LogicNot: tacop.TacUnaryOp.LogicNot,
        }[expr.op]
        operand_temp = expr.operand.getattr("val")
        if isinstance(expr.operand, Identifier):
            operand_temp = mv.freshTemp()
            mv.visitLoadWord(operand_temp, expr.operand.getattr("address"), 0, expr.operand)
        expr.setattr("val", mv.visitUnary(op, operand_temp))

    def visitBinary(self, expr: Binary, mv: TACFuncEmitter) -> None:
        expr.lhs.accept(self, mv)
        expr.rhs.accept(self, mv)

        op = {
            node.BinaryOp.Add: tacop.TacBinaryOp.ADD,
            node.BinaryOp.Sub: tacop.TacBinaryOp.SUB,
            
            node.BinaryOp.Mul: tacop.TacBinaryOp.MUL,
            node.BinaryOp.Div: tacop.TacBinaryOp.DIV,
            node.BinaryOp.Mod: tacop.TacBinaryOp.MOD,
            
            node.BinaryOp.LogicOr: tacop.TacBinaryOp.OR,
            node.BinaryOp.LogicAnd: tacop.TacBinaryOp.AND,
            
            node.BinaryOp.EQ: tacop.TacBinaryOp.EQU,
            node.BinaryOp.NE: tacop.TacBinaryOp.NEQ,
            
            node.BinaryOp.LT: tacop.TacBinaryOp.SLT,
            node.BinaryOp.GT: tacop.TacBinaryOp.SGT,
            node.BinaryOp.LE: tacop.TacBinaryOp.LEQ,
            node.BinaryOp.GE: tacop.TacBinaryOp.GEQ,
            # You can add binary operations here.
        }[expr.op]
        lhs_temp = expr.lhs.getattr("val")
        rhs_temp = expr.rhs.getattr("val")
        # print("binary", type(expr.lhs), type(expr.rhs))
        if isinstance(expr.lhs, Identifier):
            # print("ident")
            lhs_temp = mv.freshTemp()
            mv.visitLoadWord(lhs_temp, expr.lhs.getattr("address"), 0, expr.lhs)
        if isinstance(expr.rhs, Identifier):
            rhs_temp = mv.freshTemp()
            mv.visitLoadWord(rhs_temp, expr.rhs.getattr("address"), 0, expr.rhs)
        expr.setattr(
            "val", mv.visitBinary(op, lhs_temp, rhs_temp)
        )

    def visitCondExpr(self, expr: ConditionExpression, mv: TACFuncEmitter) -> None:
        """
        1. Refer to the implementation of visitIf and visitBinary.
        """
        expr.cond.accept(self, mv)
        skipLabel = mv.freshLabel()
        exitLabel = mv.freshLabel()
        value = mv.freshTemp()
        mv.visitCondBranch(
            tacop.CondBranchOp.BEQ, expr.cond.getattr("val"), skipLabel
        )
        expr.then.accept(self, mv)
        mv.visitAssignment(value, expr.then.getattr("val"))
        mv.visitBranch(exitLabel)
        mv.visitLabel(skipLabel)
        
        expr.otherwise.accept(self, mv)
        mv.visitAssignment(value, expr.otherwise.getattr("val"))
        mv.visitLabel(exitLabel)
        expr.setattr( "val", value)
        # raise NotImplementedError

    def visitIntLiteral(self, expr: IntLiteral, mv: TACFuncEmitter) -> None:
        expr.setattr("val", mv.visitLoad(expr.value))
        # address = mv.freshTemp()
        # mv.visitAlloc(address, 4)
        # mv.visitSaveWord(expr.getattr("val"), address, 0)
        # expr.setattr("address", address)
        
    def visitCall(self, call: Call, mv: TACFuncEmitter) -> None:
        arg_list = []
        for arg in call.argument_list:
            arg.accept(self, mv)
            arg_list.append(arg.getattr("val"))
            # print(arg.getattr("val"), type(arg.getattr("val")))
        for arg in call.argument_list:
            val = arg.getattr("val")
            mv.visitParam(val)
        dst = mv.freshTemp()
        mv.visitCall(dst, FuncLabel(call.ident.value), arg_list)
        call.setattr("val", dst)
        
    def visitArrayElement(self, array_element: ArrayElement, mv: TACFuncEmitter) -> None:
        symbol = array_element.getattr("symbol")
        # ident_symbol = array_element.getattr("symbol")
        # symbol.temp  = ident_symbol.temp
        address = mv.freshTemp()
        # print(address)
        val = mv.freshTemp()
        if symbol.isGlobal:
            mv.visitLoadSymbol(address, symbol)
        else:
            mv.visitAssignment(address, symbol.temp)
            
        decaf_type = symbol.type # BuiltIn_type or Array_type
        for idx in array_element.indexes:
            # print(idx)
            idx.accept(self, mv)
            mv.visitArrayElement(address, idx.getattr("val"), decaf_type.base.size)
            decaf_type = decaf_type.base
        
        mv.visitLoadWord(val, address, 0)
        # print(val, address)
        array_element.setattr("val", val)
        
