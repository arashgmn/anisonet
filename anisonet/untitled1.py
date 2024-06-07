class Configer(object):
    
    def __init__(self, pops_cfg = {}):
        self.pops_cfg = pops_cfg
        
    # def __new__(cls):
    #     new =  object.__new__(cls)
    #     new.__init__()
    #     return new
    
    def add_pop(self, name, gs):
        pop_cfg = {}
        pop_cfg['gs'] = gs
        
        if name in self.pops_cfg:
            self.pops_cfg[name].update(pop_cfg)
        else:
            self.pops_cfg[name] = pop_cfg
        
    def get(self):
        return self.pops_cfg
    
import copy

def get_config(name):    
    if name == 'I':
        c = Configer()
        print('internal c:', c)
        c.add_pop('I', 100)
        
    elif name == 'IE':
        c = get_config('I')
        # c = copy.copy(get_config('I'))
        c.add_pop('E', 200)
    
    else:
        raise
        
    return c


# if __name__ == '__main__':
    
# test 1
a = get_config('I')
print(a, id(a), a.get())
# <__main__.Configer object at 0x7fa83615c580> 140360438629760 {'I': {'gs': 100}}


# test 2
b = get_config('IE')
print(b, id(b), b.get())
# <__main__.Configer object at 0x7fa83603f610> 140360437462544 {'I': {'gs': 100}, 'E': {'gs': 200}}


# test 3
c = get_config('I')
print(c, id(c), c.get())
# <__main__.Configer object at 0x7fa83603c280> 140360437449344 {'I': {'gs': 100}, 'E': {'gs': 200}}