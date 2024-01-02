from typing import Optional

from frontend.symbol.symbol import Symbol

from .scope import Scope

class ScopeStack:
    def __init__(self, globalscope: Scope) -> None:
        self.globalscope = globalscope
        self.stack = [globalscope]
        self.loop = 0
        
    def current_scope(self):
        # 当前的作用域
        return self.stack[-1]
    
    def push_scope(self, scope: Scope):
        self.stack.append(scope)
        
    def pop_scope(self):
        self.stack.pop()
        
    # To declare a symbol.
    def declare(self, symbol: Symbol) -> None:
        current = self.current_scope()
        current.declare(symbol)
    
    # To declare a global symbol
    def declare_global(self, symbol: Symbol) -> None:
        global_scope = self.stack[0]
        global_scope.declare(symbol) 

    # To check if this is a global scope.
    def isGlobalScope(self) -> bool:
        current = self.current_scope()
        return current.isGlobalScope()
    
    # To get a symbol if declared in the scope
    def lookup(self, name: str) -> Optional[Symbol]:
        current = self.current_scope()
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].lookup(name) is not None:
                return self.stack[index].lookup(name)
        return None
    
    def lookup_current(self, name: str) -> Optional[Symbol]:
        current = self.current_scope()
        return current.lookup(name)
    
    def print_scopes(self):
        for i in self.stack:
            print(i.symbols)
            
    def begin_loop(self):
        self.loop += 1
    
    def end_loop(self):
        self.loop -= 1
        
    def is_loop(self):
        return self.loop