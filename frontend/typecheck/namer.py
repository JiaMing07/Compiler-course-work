from typing import Protocol, TypeVar, cast

from frontend.ast.node import Node, NullType
from frontend.ast.tree import *
from frontend.ast.visitor import RecursiveVisitor, Visitor
from frontend.scope.globalscope import GlobalScope
from frontend.scope.scope import Scope, ScopeKind
from frontend.scope.scopestack import ScopeStack
from frontend.symbol.funcsymbol import FuncSymbol
from frontend.symbol.symbol import Symbol
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from frontend.type.type import DecafType
from utils.error import *
from utils.riscv import MAX_INT

"""
The namer phase: resolve all symbols defined in the abstract 
syntax tree and store them in symbol tables (i.e. scopes).
"""


class Namer(Visitor[ScopeStack, None]):
    def __init__(self) -> None:
        pass

    # Entry of this phase
    def transform(self, program: Program) -> Program:
        # print(program)
        # Global scope. You don't have to consider it until Step 6.
        program.globalScope = GlobalScope
        ctx = ScopeStack(program.globalScope)

        program.accept(self, ctx)
        return program

    def visitProgram(self, program: Program, ctx: ScopeStack) -> None:
        # Check if the 'main' function is missing
        if not program.hasMainFunc():
            raise DecafNoMainFuncError

        for func in program.functions().values():
            func.accept(self, ctx)

    def visitFunction(self, func: Function, ctx: ScopeStack) -> None:
        func_sym = ctx.lookup(func.ident.value)
        if func_sym is not None and not func_sym.isFunc:
            raise DecafDeclConflictError(func.ident.value)
        
        func_sym = FuncSymbol(func.ident.value, func.ret_t.type, ctx.current_scope())
        params_name = [param.ident.value for param in func.params]
        # print(params_name)
        single_list = list(set(params_name))
        if len(single_list) != len(params_name):
            raise DecafDeclConflictError("param")
        for param in func.params:
            func_sym.addParaType(param.var_t)
        func.setattr("symbol", func_sym)
        
        scope = Scope(ScopeKind.FORMAL)
        ctx.declare(func_sym)
        ctx.push_scope(scope)
        for param in func.params:
            param.accept(self, ctx)
        # print(ctx.current_scope().symbols)
        func.body.accept(self, ctx)
        ctx.pop_scope()
        
    def visitParameter(self, param: Parameter, ctx: ScopeStack) -> None:
        new_varsymbol = VarSymbol(param.ident.value, param.var_t.type)
        ctx.declare(new_varsymbol)
        param.setattr("symbol", new_varsymbol)
        
    def visitCall(self, call: Call, ctx: ScopeStack) ->None:
        func_sym: FuncSymbol = ctx.lookup(call.ident.value)
        if func_sym is None or not func_sym.isFunc:
            raise DecafUndefinedFuncError(call.ident.value)
        if len(call.argument_list) != len(func_sym.para_type):
            raise DecafBadFuncCallError(f"{call.ident.value}'s params length error")
        for arg in call.argument_list:
            # print("arg", type(arg), arg.op, arg.lhs, arg.rhs)
            arg.accept(self, ctx)
            # print(arg.type, arg.getattr("symbol"), type(arg))
        for idx, arg in enumerate(call.argument_list):
            # print(func_sym.para_type[idx].type, type(arg.type))
            # if arg.type.name != func_sym.para_type[idx].type.name:
            # print(arg.type)
            if not isinstance(arg.type, type(func_sym.para_type[idx].type)):
                # print(isinstance(arg.type,type(func_sym.para_type[idx].type)))
                raise DecafBadFuncCallError(f"{call.ident.value}'s arg {arg.name} type error, arg.type {arg.type}, func_sym.para_type[idx].type {func_sym.para_type[idx].type}")

        

    def visitBlock(self, block: Block, ctx: ScopeStack) -> None:
        # print("block", len(ctx.stack))
        current_scope = ctx.current_scope()
        scope = Scope(ScopeKind.LOCAL)
        if current_scope.kind == ScopeKind.FORMAL:
            symbols = current_scope.symbols
            scope.symbols.update(symbols)
        ctx.push_scope(scope)
        # print("if", ctx.current_scope().symbols, ctx.stack)
        # print("curr", ctx.current_scope().symbols, ctx.stack)
        for child in block:
            # print(type(child), ctx.stack,ctx.current_scope().symbols)
            child.accept(self, ctx)
        ctx.pop_scope()

    def visitReturn(self, stmt: Return, ctx: ScopeStack) -> None:
        stmt.expr.accept(self, ctx)

    """
    def visitFor(self, stmt: For, ctx: Scope) -> None:

    1. Open a local scope for stmt.init.
    2. Visit stmt.init, stmt.cond, stmt.update.
    3. Open a loop in ctx (for validity checking of break/continue)
    4. Visit body of the loop.
    5. Close the loop and the local scope.
    """

    def visitIf(self, stmt: If, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        stmt.then.accept(self, ctx)

        # check if the else branch exists
        if not stmt.otherwise is NULL:
            stmt.otherwise.accept(self, ctx)

    def visitWhile(self, stmt: While, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        ctx.begin_loop()
        stmt.body.accept(self, ctx)
        ctx.end_loop()
        
    def visitDoWhile(self, stmt: DoWhile, ctx: ScopeStack) -> None:
        ctx.begin_loop()
        stmt.body.accept(self, ctx)
        ctx.end_loop()
        stmt.cond.accept(self, ctx)
        
    def visitFor(self, stmt: For, ctx: ScopeStack) -> None:
        scope = Scope(ScopeKind.LOCAL)
        ctx.push_scope(scope)
        stmt.init.accept(self,ctx)
        stmt.cond.accept(self, ctx)
        stmt.incr.accept(self, ctx)
        ctx.begin_loop()
        stmt.body.accept(self, ctx)
        ctx.end_loop()
        ctx.pop_scope()

    def visitBreak(self, stmt: Break, ctx: ScopeStack) -> None:
        """
        You need to check if it is currently within the loop.
        To do this, you may need to check 'visitWhile'.

        if not in a loop:
            raise DecafBreakOutsideLoopError()
        """
        if not ctx.is_loop():
            raise DecafBreakOutsideLoopError()
        # raise NotImplementedError

    
    def visitContinue(self, stmt: Continue, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBreak.
        """
        if not ctx.is_loop():
            raise DecafBreakOutsideLoopError()

    def visitDeclaration(self, decl: Declaration, ctx: ScopeStack) -> None:
        """
        1. Use ctx.lookup to find if a variable with the same name has been declared.
        2. If not, build a new VarSymbol, and put it into the current scope using ctx.declare.
        3. Set the 'symbol' attribute of decl.
        4. If there is an initial value, visit it.
        """
        sym = ctx.lookup_current(decl.ident.value)
        # print(sym, ctx.current_scope().symbols)
        if sym != None:
            raise DecafDeclConflictError(decl.ident.value)
        else:
            new_varsymbol = VarSymbol(decl.ident.value, decl.var_t.type)
            ctx.declare(new_varsymbol)
            decl.setattr("symbol", new_varsymbol)
            if decl.init_expr is not NULL:
                decl.init_expr.accept(self, ctx)
        # raise NotImplementedError

    def visitAssignment(self, expr: Assignment, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        sym = ctx.lookup(expr.lhs.value)
        if sym is None:
            raise DecafUndefinedVarError(expr.lhs.value)
        else:
            expr.lhs.accept(self, ctx)
            expr.rhs.accept(self, ctx)
        expr.type = expr.lhs.type
        # raise NotImplementedError

    def visitUnary(self, expr: Unary, ctx: ScopeStack) -> None:
        expr.operand.accept(self, ctx)
        expr.type = expr.operand.type

    def visitBinary(self, expr: Binary, ctx: ScopeStack) -> None:
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)
        expr.type = expr.lhs.type

    def visitCondExpr(self, expr: ConditionExpression, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.cond.accept(self, ctx)
        expr.then.accept(self, ctx)
        expr.otherwise.accept(self, ctx)
        expr.type = expr.then.type
        # raise NotImplementedError

    def visitIdentifier(self, ident: Identifier, ctx: ScopeStack) -> None:
        """
        1. Use ctx.lookup to find the symbol corresponding to ident.
        2. If it has not been declared, raise a DecafUndefinedVarError.
        3. Set the 'symbol' attribute of ident.
        """
        sym = ctx.lookup(ident.value)
        if sym is None:
            raise DecafUndefinedVarError(ident.value)
        else:
            # print("set")
            ident.setattr("symbol", sym)
        # print(ident.value, sym.type, sym)
        ident.type = sym.type
        # print(ident.type)
        # raise NotImplementedError

    def visitIntLiteral(self, expr: IntLiteral, ctx: ScopeStack) -> None:
        value = expr.value
        if value > MAX_INT:
            raise DecafBadIntValueError(value)
