from backend.dataflow.basicblock import BasicBlock
from backend.dataflow.cfg import CFG
from queue import Queue
from utils.tac.tacinstr import *
from backend.dataflow.loc import Loc
from utils.label.label import Label
from utils.label.blocklabel import BlockLabel



class Dominate:
    """
    通过 CFG 构造
    在构造函数中首先计算一个结点的支配集合（self.dominates），再计算它的直接支配节点(self.idoms)
    得到直接支配节点后可以构建一个支配树(self.tree_links)，再计算一个结点的支配边界(self.dfs)
    然后调用 insert_phi 函数插入 phi 函数
    最后调用 rename 函数进行变量重命名
    """
    def __init__(self, cfg: CFG, temp_id, func_name, label_id) -> None:
        self.nodes = cfg.nodes
        self.edges = cfg.edges
        self.reachable_blocks = cfg.reachable_blocks
        self.new_temp_id = temp_id
        self.var_dict = None
        self.var_address = None
        self.func_name = func_name
        self.label_id = label_id
        
        # 为每一个可以到达的块设置一个 label，为 Phi 函数做准备
        for block in self.reachable_blocks:
            if self.nodes[block].label is None:
                self.label_id += 1
                self.nodes[block].label = BlockLabel(str(self.label_id))
                
        
        for edge in cfg.edges:
            if edge[0] not in self.reachable_blocks or edge[1] not in self.reachable_blocks:
                self.edges.remove(edge)

        self.links = []

        for i in range(len(self.nodes)):
            self.links.append((set(), set()))

        for (u, v) in self.edges:
            self.links[u][1].add(v)
            self.links[v][0].add(u)
            

        self.dominates = []
        # 节点本身属于它自己的支配集合
        for i in range(len(self.nodes)):
            self.dominates.append(set())
            self.dominates[i].add(i)
            
        sum = 0
        # 循环迭代知道所有节点的支配集合都不再变化
        while True:
            sum += 1
            flag = True
            for i, link in enumerate(self.links):
                s = set()
                idx = 0
                # 计算所有前驱节点支配集合的并集再与自己求交
                for pred in link[0]:
                    if idx == 0:
                        s = self.dominates[pred]
                    elif idx < i:
                        s = s.intersection(self.dominates[pred])
                    idx += 1
                if self.dominates[i] != set.union(s,set({i})):
                    flag = False  
                self.dominates[i] = set.union(self.dominates[i], s)
            if flag == True:
                break
            
        # 去除本身，得到严格支配集合
        for i, dom in enumerate(self.dominates):
            dom.remove(i)
        self.idoms = []
        # 找到严格支配 n（idx），且不严格支配任何严格支配 n（idx） 的节点的节点
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
    
        # 构建支配树
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
        
        # 再把 self.dominates 变成支配集合（加上自己）
        for i, dom in enumerate(self.dominates):
            dom.add(i)
            
        self.dfs = []
        for i in range(len(self.nodes)):
            self.dfs.append((set()))
        # 找到所有 v，使得 x 支配 v 的前驱节点，x 不严格支配 v   
        for (u, v) in self.edges:
            x = u
            while x not in self.dominates[v]:
                self.dfs[x] = set.union(self.dfs[x], set({v}))
                x = self.idoms[x]
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
        self.insert_phi()
        self.rename()
        
    def insert_phi(self):
        """
        插入 phi 函数
        https://szp15.com/post/how-to-construct-ssa/#%E4%BD%BF%E7%94%A8%E6%94%AF%E9%85%8D%E8%BE%B9%E7%95%8C%E5%AF%BB%E6%89%BE%CF%95%E5%87%BD%E6%95%B0%E9%9C%80%E8%A6%81%E7%9A%84%E5%9C%B0%E6%96%B9
        """
        temp_id = self.new_temp_id
        var_dict = {}
        var_address = []
        # 首先计算每个变量的 worklist（在哪一个块中被复制），
        # 由于局部变量的地址是唯一的，所以选用 tuple(address, ident.value) 的形式作为 key 保存
        for idx_block, node in enumerate(self.nodes):
            for l in node.locs:
                if isinstance(l.instr, Alloc) or isinstance(l.instr, SaveWord):
                    if l.instr.ident != None:
                        if l.instr.ident.getattr("address") not in var_address:
                            var_address.append(l.instr.ident.getattr("address"))
                            var_dict[(l.instr.ident.getattr("address"), l.instr.ident.value)] = [idx_block]
                        else:
                            var_dict[(l.instr.ident.getattr("address"), l.instr.ident.value)].append(idx_block)
        # 对每一个变量
        for var in var_dict.keys():
            worklist = list(set(var_dict[var]))
            visited = [False for i in range(len(self.nodes))]
            placed = [False for i in range(len(self.nodes))]
            while len(worklist) != 0: # 当 worklist 不为空时循环
                x = worklist.pop(-1) # 弹出最后一个节点
                for y in self.dfs[x]:
                    # 对所有 x 的支配边界的元素如果还没有 phi 指令就插入一条
                    if placed[y] == False:
                        placed[y] = True
                        self.nodes[y].locs.insert(0, Loc(Phi(Temp(self.new_temp_id), var)))
                        self.new_temp_id += 1
                        if visited[y] == False: # 如果还没有访问过这一块就将这一块添加进 worklist 中
                            visited[y] = True
                            worklist.append(y)
        self.var_dict = var_dict
        self.var_address = var_address  
        # print("-------------------")
        # for idx_block, node in enumerate(self.nodes):
        #     # print(node.kind)
        #     if node.label is not None:
        #         if idx_block == 0:
        #             print(str(self.func_name)+":")
        #         print(str(node.label) + ":")
        #     for l in node.locs:
        #         print("    "  + str(l.instr))
                
        # print("-------------------")
        
    def rename(self):
        # 重命名阶段，此函数内为准备阶段，search 中为具体实现
        stack = {}
        counters = self.new_temp_id
        for key in self.var_dict.keys():
            stack[key] = []
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
        # 参考 https://buaa-se-compiling.github.io/miniSysY-tutorial/challenge/mem2reg/help.html
        # 参考 https://szp15.com/post/how-to-construct-ssa
        remove_index = []
        stack_push = {}
        for key in self.var_dict.keys():
            stack_push[key] = 0
        for idx, l in enumerate(node.locs):
            if isinstance(l.instr, Alloc) and l.instr.ident is not None:
                # 记录要删除的指令位置
                remove_index.append(l)
            elif isinstance(l.instr, LoadWord) and l.instr.ident is not None:
                # 对 Load 指令，将所有其他指令中对该指令的使用替换为对该变量到达定义的使用，记录要删除的指令位置
                dst = l.instr.dsts[0]
                key = (l.instr.ident.getattr("address"), l.instr.ident.value)
                for i in range(idx + 1, len(node.locs)):
                    for src_idx, src in enumerate(node.locs[i].instr.srcs):
                        if src == dst:
                            node.locs[i].instr.srcs[src_idx].index = stack[key][-1].index
                remove_index.append(l)
            elif isinstance(l.instr, SaveWord) and l.instr.ident is not None:
                # 对 Save 指令，更新 stack （更新该变量的到达定义），记录要删除的指令位置
                key = (l.instr.ident.getattr("address"), l.instr.ident.value)
                counters += 1
                stack[key].append(l.instr.dsts[0])
                stack_push[key] += 1
                remove_index.append(l)
            elif isinstance(l.instr, Phi): 
                # 对 Phi 指令，更新 stack （更新该变量的到达定义）并将所有其他指令中对该指令的使用替换为对该变量到达定义的使用
                key = l.instr.ident
                stack[key].append(l.instr.dsts[0])
                stack_push[key] += 1
                dst = l.instr.dsts[0]
                for i in range(idx + 1, len(node.locs)):
                    for src_idx, src in enumerate(node.locs[i].instr.srcs):
                        if src == dst:
                            node.locs[i].instr.srcs[src_idx].index = stack[key][-1].index   
        # 删除 Alloc LoadWord SaveWord 指令
        for remove_idx in remove_index:
            node.locs.remove(remove_idx)
        # 维护该基本块的所有后继基本块中的 phi 指令，将对应来自此基本块的值设为对应变量的到达定义
        for var in self.var_dict.keys():
            for succ in self.links[idx_block][1]:
                for idx_l, l in enumerate(self.nodes[succ].locs):
                    if isinstance(l.instr, Phi) and l.instr.ident == var:
                        label = node.label
                        try:
                            tmp = stack[var][-1]
                            l.instr.add_label(label)
                            l.instr.add_src(stack[var][-1], label)
                        except:
                            pass
        # 处理所有的孩子节点
        for child in self.tree_links[idx_block][1]:
            self.search(child, self.nodes[child], stack, counters)
        # 将本块中的新定义退栈
        for key in self.var_dict.keys():
            if stack_push[key] != 0:
                stack[key] = stack[key][:-stack_push[key]]