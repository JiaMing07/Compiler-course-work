from backend.dataflow.basicblock import BasicBlock
from backend.dataflow.cfg import CFG
from queue import Queue
from utils.tac.tacinstr import *
from backend.dataflow.loc import Loc
from utils.label.label import Label
from utils.label.blocklabel import BlockLabel



class Dominate:
    def __init__(self, cfg: CFG, temp_id, func_name, label_id) -> None:
        self.nodes = cfg.nodes
        self.edges = cfg.edges
        self.reachable_blocks = cfg.reachable_blocks
        self.new_temp_id = temp_id
        self.var_dict = None
        self.var_address = None
        self.func_name = func_name
        self.label_id = label_id
        
        for block in self.reachable_blocks:
            if self.nodes[block].label is None:
                self.label_id += 1
                self.nodes[block].label = BlockLabel(str(self.label_id))
                
        
        for edge in cfg.edges:
            if edge[0] not in self.reachable_blocks or edge[1] not in self.reachable_blocks:
                self.edges.remove(edge)
        # print("edges", self.edges)

        self.links = []

        for i in range(len(self.nodes)):
            self.links.append((set(), set()))

        for (u, v) in self.edges:
            self.links[u][1].add(v)
            self.links[v][0].add(u)
            
        # print("links", self.links)

        self.dominates = []
        
        for i in range(len(self.nodes)):
            self.dominates.append(set())
            self.dominates[i].add(i)
        # print(self.dominates)
            
        sum = 0
        while True:
            sum += 1
            # print("sum", sum)
            flag = True
            for i, link in enumerate(self.links):
                s = set()
                idx = 0
                for pred in link[0]:
                    if idx == 0:
                        # print(i, pred)
                        s = self.dominates[pred]
                        # print(pred, self.dominates[pred])
                    elif idx < i:
                        s = s.intersection(self.dominates[pred])
                    idx += 1
                if self.dominates[i] != set.union(s,set({i})):
                    flag = False  
                self.dominates[i] = set.union(self.dominates[i], s)
            if flag == True:
                break
            
        # print("dominates",self.dominates)
       
        for i, dom in enumerate(self.dominates):
            dom.remove(i)
        # print("strict dominates",self.dominates)
        self.idoms = []
        
        for idx, dominate in enumerate(self.dominates):
            idom = None
            for i in dominate:
                if len(dominate) == 1:
                    idom = i
                for j in dominate:
                    if i != j:
                        if i not in self.dominates[j]:
                            idom = i
            self.idoms.append(idom)
        # print(self.idoms)
    
        
        self.tree_edges = []

        for i, tree_pred in enumerate(self.idoms):
            if tree_pred != None:
                self.tree_edges.append((tree_pred, i))
        # print(self.tree_edges)
        
        self.tree_links = []

        for i in range(len(self.nodes)):
            self.tree_links.append((set(), set()))

        for (u, v) in self.tree_edges:
            self.tree_links[u][1].add(v)
            self.tree_links[v][0].add(u)
        # print("tree links:",self.tree_links)
        
        for i, dom in enumerate(self.dominates):
            dom.add(i)
            
        self.dfs = []
        for i in range(len(self.nodes)):
            self.dfs.append((set()))
            
        for (u, v) in self.edges:
            x = u
            while x not in self.dominates[v]:
                # print(x, self.dfs)
                self.dfs[x] = set.union(self.dfs[x], set({v}))
                x = self.idoms[x]
        # print(self.dfs)
        self.insert_phi()
        self.rename()
        
    def insert_phi(self):
        temp_id = self.new_temp_id
        var_dict = {}
        var_address = []
        for idx_block, node in enumerate(self.nodes):
            for l in node.locs:
                if isinstance(l.instr, Alloc) or isinstance(l.instr, SaveWord):
                    if l.instr.ident != None:
                        if l.instr.ident.getattr("address") not in var_address:
                            var_address.append(l.instr.ident.getattr("address"))
                            var_dict[(l.instr.ident.getattr("address"), l.instr.ident.value)] = [idx_block]
                        else:
                            var_dict[(l.instr.ident.getattr("address"), l.instr.ident.value)].append(idx_block)
        for var in var_dict.keys():
            worklist = var_dict[var]
            visited = [False for i in range(len(self.nodes))]
            placed = [False for i in range(len(self.nodes))]
            while len(worklist) != 0:
                x = worklist.pop(0)
                for y in self.dfs[x]:
                    if placed[y] == False:
                        placed[y] = True
                        
                        phi = Phi(Temp(self.new_temp_id), var)
                        self.nodes[y].locs.insert(0, Loc(Phi(Temp(self.new_temp_id), var)))
                        # print("label: ", self.nodes[y].label)
                        self.new_temp_id += 1
        self.var_dict = var_dict
        self.var_address = var_address    
        
    def rename(self):
        stack = {}
        counters = self.new_temp_id
        for key in self.var_dict.keys():
            stack[key] = []
        return_idom = None
        return_key = None
        for idx_block, node in enumerate(self.nodes):
            break_flag = False
            for idx, l in enumerate(node.locs):
                if isinstance(l.instr, Return):
                    flag = False
                    for idx_, l_ in enumerate(node.locs):
                        if isinstance(l_.instr, Phi) and l_.instr.ident[0] == l.instr.value:
                            flag = True
                    if not flag:
                        # print("return", node.locs[idx-1].instr.dsts[0], l.instr.value)
                        if idx > 0 and isinstance(node.locs[idx-1].instr, LoadWord) and l.instr.value == node.locs[idx-1].instr.dsts[0]:
                            return_idom = self.idoms[idx_block]
                            return_key = (node.locs[idx-1].instr.srcs[0], node.locs[idx-1].instr.ident.value)
                            break_flag = True
                            break
            if break_flag:
                break
        return_value = None   
        self.search(0, self.nodes[0], stack, counters) 
                            
                    
        print("-------------------")
        for idx_block, node in enumerate(self.nodes):
            # print(node.kind)
            if node.label is not None:
                if idx_block == 0:
                    print(str(self.func_name)+":")
                print(str(node.label) + ":")
            for l in node.locs:
                print("    "  + str(l.instr))
                
        print("-------------------")
        
    def search(self, idx_block, node, stack, counters):
        remove_index = []
        stack_push = {}
        for key in self.var_dict.keys():
            stack_push[key] = 0
        for idx, l in enumerate(node.locs):
            # print("type: ", l.instr, idx)
            if isinstance(l.instr, Alloc) and l.instr.ident is not None:
                remove_index.append(l)
            elif isinstance(l.instr, LoadWord) and l.instr.ident is not None:
                dst = l.instr.dsts[0]
                key = (l.instr.ident.getattr("address"), l.instr.ident.value)
                # node.locs.remove(l)
                for i in range(idx + 1, len(node.locs)):
                    for src_idx, src in enumerate(node.locs[i].instr.srcs):
                        if src == dst:
                            # print("src1:", node.locs[i].instr, dst.index, node.locs[i].instr.srcs[src_idx].index)
                            node.locs[i].instr.srcs[src_idx].index = stack[key][-1].index
                            # print("src2:", node.locs[i].instr, dst.index, "stack:", stack[key][-1])
                remove_index.append(l)
            elif isinstance(l.instr, SaveWord) and l.instr.ident is not None:
                key = (l.instr.ident.getattr("address"), l.instr.ident.value)
                # temp = Temp(counters)
                # print(l.instr.dsts[0])
                counters += 1
                stack[key].append(l.instr.dsts[0])
                stack_push[key] += 1
                # node.locs.remove(l)
                remove_index.append(l)
            elif isinstance(l.instr, Phi):
                # print("phi", l.instr.dsts[0].index)
                key = l.instr.ident
                stack[key].append(l.instr.dsts[0])
                stack_push[key] += 1
                dst = l.instr.dsts[0]
                for i in range(idx + 1, len(node.locs)):
                    for src_idx, src in enumerate(node.locs[i].instr.srcs):
                        if src == dst:
                            # print("src1:", node.locs[i].instr, dst.index, node.locs[i].instr.srcs[src_idx].index)
                            node.locs[i].instr.srcs[src_idx].index = stack[key][-1].index
                            # print("src2:", node.locs[i].instr, dst.index, "stack:", stack[key][-1])
            # elif isinstance(l.instr,Return):
            #     if return_value is not None:
            #         # print("return change", return_value)
            #         l.instr.value = return_value
        for remove_idx in remove_index:
            node.locs.remove(remove_idx)
        for var in self.var_dict.keys():
            for succ in self.links[idx_block][1]:
                for idx_l, l in enumerate(self.nodes[succ].locs):
                    if isinstance(l.instr, Phi) and l.instr.ident == var:
                        label = node.label
                        l.instr.add_label(label)
                        l.instr.add_src(stack[var][-1], label)
        # print(idx_block, stack_push)
        # print(idx_block,stack)
        for child in self.tree_links[idx_block][1]:
            # print(idx_block, child)
            self.search(child, self.nodes[child], stack, counters)
        # print(idx_block,stack)
        for key in self.var_dict.keys():
            if stack_push[key] != 0:
                stack[key] = stack[key][:-stack_push[key]]
        # print(idx_block, stack)