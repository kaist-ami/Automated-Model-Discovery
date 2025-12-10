from lark import Lark, Transformer, v_args

from GPy_ABCD.Kernels.baseKernels import *
from GPy_ABCD.KernelExpansion.grammar import *
from GPy_ABCD.Util.dataAndPlottingUtil import *
from GPy_ABCD.Util.modelUtil import *
from GPy_ABCD.Kernels import changeOperators as _Cs
from GPy_ABCD.Kernels import baseKernels
import GPy

import re
import random
import numpy as np

def is_float(value):
    # 정규식을 사용하여 숫자 형식인지 확인
    value = str(value)
    float_pattern = re.compile(r'^-?\d+(\.\d+)?$')
    return bool(float_pattern.match(value))

class CFGParamDictTransformer(Transformer):
    def __init__(self, dim=1):
        self.dim = dim
        
    def debug(self, items):
        return items[0]
    
    @v_args(inline=True)
    def add(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) + float(right)
        import pdb; pdb.set_trace()
        left.update(right)
        return left
    
    @v_args(inline=True)
    def subtract(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) - float(right)
        left.update(right)
        return left

    @v_args(inline=True)
    def prod(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) * float(right)
        left.update(right)
        return left
    
    @v_args(inline=True)
    def divide(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) / float(right)
        left.update(right)
        return left
    
    @v_args(inline=True)
    def power(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) ** float(right)
    
    def change_function(self, items):
        k1 = items[0]
        k2 = items[1]
        k1.update(k2)
        return k1
        #### location todo
        
    def change_function2(self, items):
        k1 = items[0]
        k2 = items[1]
        k1.update(k2)
        return k1
        #### location todo
    
    def sum_function(self, items):
        # "+" 연산자를 SUM 함수로 변환
        k1 = items[0]
        k2 = items[1]
            
        k1.update(k2)
        return k1

    def product_function(self, items):
        # "*" 연산자를 PRODUCT 함수로 변환
        k1 = items[0]
        k2 = items[1]        
        k1.update(k2)
        return k1
        # return items
        
    def error(self, value):
        return 'NUM'
    
    def param_block(self, items):
        return dict(items[0])
    
    def param_list(self, items):
        paramdict = dict(items)
        for k in paramdict.copy():
            if paramdict[k]=='NUM': paramdict.pop(k)
        return paramdict  # 파라미터 리스트를 딕셔너리로 변환
        
    def params(self, items):
        return dict(items)
            
    def param(self, items):
        # import pdb; pdb.set_trace()
            
        if not is_float(items[1]):
            return (str(items[0]), 'NUM')
        
        else:
            return (str(items[0]), float(items[1]))
        # return (str(items[0]), str(items[1]) if str(items[1]) == "unknown" else float(items[1]))

    def _get_value(self, value):
        if not is_float(value): return "NUM"
        return float(value)

    def argument(self, items):
        name = str(items[0])
        #### 좀 이상해도일단이렇게...
        if len(items) == 1: params = {}
        else: params = items[1:][0]
        param_dict = {}
        
        if name == "LIN":        
            if 'offset' in params and is_float(params['offset']): 
                param_dict['LIN'] = {'offset': params['offset']}
            
        elif name == "PER":
            param_dict['PER'] = {}
            if 'period' in params and is_float(params['period']):
                param_dict['PER']['period'] = params['period'] * 2
                
            if 'lengthscale' in params and is_float(params['lengthscale']):
                param_dict['PER']['lengthscale'] = params['lengthscale']
                
            if len(param_dict['PER'])==0: param_dict.pop('PER')
            
        elif name == "SE" or name == "RBF":
            param_dict['SE'] = {}
            if 'lengthscale' in params and is_float(params['lengthscale']):
                param_dict['SE']['lengthscale'] = params['lengthscale']
            
            if len(param_dict['SE'])==0: param_dict.pop('SE')
            
        return param_dict
        

class CFGWPARAMTransformer(Transformer):
    def __init__(self,  dim=1):
        self.dim = dim
        
    def debug(self, items):
        # import pdb; pdb.set_trace()
        return items[0]
    
    @v_args(inline=True)
    def add(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) + float(right)
        return left + right
    
    @v_args(inline=True)
    def subtract(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) - float(right)
        return left - right

    @v_args(inline=True)
    def prod(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) * float(right)
        return left * right
    
    @v_args(inline=True)
    def divide(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) / float(right)
        return left / right
    
    @v_args(inline=True)
    def power(self, left, right):
        # import pdb; pdb.set_trace()
        if is_float(left) and is_float(right):
            return float(left) ** float(right)
    
    def change_function(self, items):
        k1 = items[0]
        k2 = items[1]
        if type(k1)==list:
            k1 = k1[0]
        if type(k2)==list:
            k2 = k2[0]
        return deep_apply(both_changes, k1, k2)[0]
        #### location todo
        
    def change_function2(self, items):
        k1 = items[0]
        k2 = items[1]
        if type(k1)==list:
            k1 = k1[0]
        if type(k2)==list:
            k2 = k2[0]
        return deep_apply(both_changes, k1, k2)[0]
        #### location todo
    
    def sum_function(self, items):
        # "+" 연산자를 SUM 함수로 변환
        k1 = items[0]
        k2 = items[1]
        if type(k1)==list:
            k1 = k1[0]
        if type(k2)==list:
            k2 = k2[0] #### 이렇게해도 되는건지체크
            
        return deep_apply(add, k1, k2)[0]

    def product_function(self, items):
        # "*" 연산자를 PRODUCT 함수로 변환
        k1 = items[0]
        k2 = items[1]
        if type(k1)==list:
            k1 = k1[0]
        if type(k2)==list:
            k2 = k2[0] #### 이렇게해도 되는건지체크
        
        return deep_apply(multiply, k1, k2)[0]
        # return items

    def argument(self, items):
        identifier = items[0]  # LIN 또는 PER 같은 인자 이름
        # params = items[1]  # 파라미터 리스트
        params = items[1].children[0] if len(items) > 1 else {}
        if identifier=='LIN':
            tmp = SumKE(['LIN'])._initialise()
            if 'slope' in params.keys() and is_float(params['slope']):
                tmp.to_kernel().variance = float(params['slope'])
                    
        elif identifier=='PER':
            tmp = SumKE(['PER', 'C'])._initialise()
            if 'period' in params.keys() and is_float(params['period']):
                tmp.to_kernel().parts[0].period = float(params['period'])

                    
        elif identifier=='SE' or identifier=='RBF':
            tmp = SumKE(['SE'])._initialise()
            if 'lengthscale' in params.keys() and is_float(params['lengthscale']):
                tmp.to_kernel().lengthscale = float(params['lengthscale'])
                
        elif identifier=='C':
            tmp = SumKE(['C'])._initialise()
            ##### TODO:
            ##### C에 대한 파라미터 처리 추가하기
            
        elif identifier=='WN':
            tmp = SumKE(['WN'])._initialise()
        else:
            raise Exception(f'{identifier} not in list')
        
        return tmp

    def param_list(self, items):
        return dict(items)  # 파라미터 리스트를 딕셔너리로 변환

    def param(self, items):
        # key = items[0]
        # value = items[1].children[0]
        # return (key, value)
        if not is_float(items[1]):
            return (str(items[0]), 'NUM')
        
        else:
            return (str(items[0]), float(items[1]))

    def IDENTIFIER(self, token):
        # if token in ['LIN', 'PER', 'SE', 'C', 'WN']:
        #     tmp = SumKE(['WN'])._initialise()
        #     return tmp
        return str(token)
        
    def NUMBER(self, token):
        return float(token)
    
    def _get_value(self, value):
        # unknown 파라미터의 경우 저장하고 초기값 반환
        if not is_float(value):
            return 1.0  # 기본값 설정
        return float(value)
    
    def error(self, value):
        import pdb; pdb.set_trace()
        return 1.0
    
    def value(self, value):
        return 1.0

def parse_kernel_name(str, dim=1):
    # EBNF 스타일로 문법 정의
    grammar = """
        start: sum_expr
            
        ?sum_expr: product_expr
            | sum_expr "+" product_expr   -> sum_function

        ?product_expr: change_expr
            | product_expr "*" change_expr       -> product_function
            
        ?change_expr: atom
            | "CP" "(" sum_expr "," sum_expr ")" param_block? -> change_function
            | "CP" "(" sum_expr "," sum_expr "," param ")" -> change_function2

        ?atom: argument                          -> debug
            | "(" sum_expr ")"                

        argument: IDENTIFIER param_block? 
        param_block: "{" param_list "}"
        param_list: param ("," param)*
        param: IDENTIFIER ":" value
        
        ?value: NAME
            | "-" value
            | value "**" value -> power
            | value "^" value -> power
            | value "/" value -> value
            | value "*" value -> value
            | value "+" value -> value
            | value "." value
            | NUMBER
            | IDENTIFIER
            | value "(" value ")"
        

        IDENTIFIER: /[A-Za-z]+/

        %import common.CNAME -> NAME
        %import common.NUMBER
        %import common.WS_INLINE
        %ignore WS_INLINE
    """
    
    # NUMBER: /-?\d+(\.\d+)?([eE][-+]?\d+)?/
    # /[0-9]+(\.[0-9]+)?/
    trans = CFGWPARAMTransformer(dim)
    l = Lark(grammar, parser='lalr', transformer=trans)
    calc = l.parse
    output = calc(str)
    kernel = trans.transform(output)
    return kernel.children[0]


############# KERNEL MODEL PARSER
class KernelTransformer(Transformer):
    def __init__(self, param_dict, dim=1):
        self.param_dict = param_dict  
        self.dim = dim

    @v_args(inline=True)
    def add(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) + float(right)
        return left + right
    
    @v_args(inline=True)
    def subtract(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) - float(right)
        return left - right

    @v_args(inline=True)
    def prod(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) * float(right)
        return left * right
    
    @v_args(inline=True)
    def divide(self, left, right):
        if is_float(left) and is_float(right):
            return float(left) / float(right)
        return left / right
    
    @v_args(inline=True)
    def power(self, left, right):
        # import pdb; pdb.set_trace()
        if is_float(left) and is_float(right):
            return float(left) ** float(right)

    @v_args(inline=True)
    def cp(self, left, right):
        ret = _Cs.ChangePointKernel(left, right, fixed_slope = True)
        return ret
        # return _Cs.ChangePointKernel(left, right, fixed_slope = True)
    @v_args(inline=True)
    def cp2(self, left, right, changepoint):
        ret = _Cs.ChangePointKernel(left, right, fixed_slope = True)
        ret.location = changepoint[1]
        return ret
    

    def kernel(self, items):
        name = str(items[0])
        #### 좀 이상해도일단이렇게...
        if len(items) == 1:
            #### no paramaeters
            params = {}
            
        else:
            params = items[1:][0]
        
        # GPy 커널 초기화
        if name == "LIN":            
            import pdb; pdb.set_trace()
            retval = baseKernels.LIN(self.dim)
            ### parameters
            # TODO: if 1, assign random
            # slope = params.get("slope", np.random.normal(0,1,1))
            bias = params.get('bias', random.random()) #### originally random.random
            bias = params.get('offset', random.random()) #### originally random.random
            
            ### assign parameters
            # retval.variance = float(slope) if is_float(slope) else np.random.normal(0,1,1)
            retval.variance = random.random()
            retval.offset = float(bias) if is_float(bias) else random.random()
            return retval
            
        elif name == "PER":
            retval = baseKernels.PER(self.dim) #+ baseKernels.C()
            
            ### parameters
            period = params.get("period", random.random())
            lengthscale = params.get("lengthscale", random.random())
            # variance = params.get('variance', random.random())
            variance = random.random()
            
            retval.period = float(period)*2 if is_float(period) else random.random()
            retval.lengthscale = float(lengthscale) if is_float(lengthscale) else random.random()
            retval.variance = float(variance) if is_float(variance) else random.random()
            return retval + baseKernels.C()
            
        elif name == "SE" or name == 'RBF':
            # if 'variance' in params and is_float(params.get("variance", random.random())):
            # variance = params.get("variance", random.random())
            variance = random.random()
            lengthscale = params.get('lengthscale', np.random.normal(5,5,1))
            return GPy.kern.RBF(input_dim=self.dim, variance=float(variance) if is_float(variance) else random.random(), lengthscale=float(lengthscale) if is_float(lengthscale) else random.random())
            
        elif name == "C":
            return GPy.kern.Bias(self.dim) ## 1 is the input_dim
        
        elif name == "WN":
            return GPy.kern.White(self.dim) ## 1 is the input_dim
        
    def param_block(self, items):
        # import pdb; pdb.set_trace()
        return dict(items[0])
        
    def params(self, items):
        return dict(items)
            
    def param(self, items):
        # import pdb; pdb.set_trace()
            
        if not is_float(items[1]):
            return (str(items[0]), 'NUM')
        
        else:
            return (str(items[0]), float(items[1]))
        # return (str(items[0]), str(items[1]) if str(items[1]) == "unknown" else float(items[1]))

    def _get_value(self, value):
        if not is_float(value): return np.random.normal(0,1,1) #### NUM 일때
        return float(value)
    
    def error(self, value):
        return np.random.normal(0,1,1)
    
    
def parse_kernel_model(str, dim=1):
    kernel_grammar = """
        ?start: expr

        ?expr: product
            | expr "+" product   -> add
            | "CP(" expr "," expr ")" -> cp
            | "CP(" expr "," expr "," param ")" -> cp2
            
        ?product: term
            | product "*" term -> prod

        ?term: kernel | "(" expr ")"

        kernel: KERNEL_NAME param_block?
        param_block: "{" params "}"
        params: param ("," param)*
        param: PARAM_NAME ":" value
            
        ?value: sum_value
        ?sum_value: product_value
            | sum_value "+" product_value   -> add
            | sum_value "-" product_value   -> subtract
            | UNKNOWN_FUNC "(" value_value ")" -> error
            | UNKNOWN_FUNC "." UNKNOWN_FUNC "(" value_value ")" -> error
            | UNKNOWN_FUNC "[" UNKNOWN_FUNC "]" -> error
        ?product_value: power_value
            | product_value "*" power_value -> prod
            | product_value "/" power_value -> divide
        ?power_value: value_value
            | value_value "^" value_value   -> power
            | value_value "**" value_value   -> power
        ?value_value: NAME
            | "-" value_value
            | NUMBER
            | IDENTIFIER
            | "(" value ")"
        
        KERNEL_NAME: /[a-zA-Z_]+/
        IDENTIFIER: /[A-Za-z]+/
        PARAM_NAME: /[a-zA-Z_]+/
        UNKNOWN_FUNC: /[a-zA-Z_]+/
        
        %import common.CNAME -> NAME
        %import common.NUMBER
        %import common.WS_INLINE
        %ignore WS_INLINE
    """

    kernel_parser = Lark(kernel_grammar, start="start")
    param_dict = {}
    kernel_transformer = KernelTransformer(param_dict, dim=dim)
    kernel_tree = kernel_parser.parse(str)
    # import pdb; pdb.set_trace()
    kernel = kernel_transformer.transform(kernel_tree)
    
    return kernel

def parse_kernel_params(str, dim=1):
    grammar = """
        ?start: sum_expr
            
        ?sum_expr: product_expr
            | sum_expr "+" product_expr   -> sum_function

        ?product_expr: change_expr
            | product_expr "*" change_expr       -> product_function
            
        ?change_expr: atom
            | "CP" "(" sum_expr "," sum_expr ")" param_block? -> change_function
            | "CP" "(" sum_expr "," sum_expr "," param ")" -> change_function2

        ?atom: argument                          -> debug
            | "(" sum_expr ")"                

        argument: IDENTIFIER param_block? 
        param_block: "{" param_list "}"
        param_list: param ("," param)*
        param: IDENTIFIER ":" value
        
        ?value: sum_value
        ?sum_value: product_value
            | sum_value "+" product_value   -> add
            | sum_value "-" product_value   -> subtract
            | UNKNOWN_FUNC "(" value_value ")" -> error
            | UNKNOWN_FUNC "." UNKNOWN_FUNC "(" value_value ")" -> error
            | UNKNOWN_FUNC "[" UNKNOWN_FUNC "]" -> error
        ?product_value: power_value
            | product_value "*" power_value -> prod
            | product_value "/" power_value -> divide
        ?power_value: value_value
            | value_value "^" value_value   -> power
            | value_value "**" value_value   -> power
        ?value_value: NAME
            | "-" value_value
            | NUMBER
            | IDENTIFIER
            | "(" value ")"
        

        KERNEL_NAME: /[a-zA-Z_]+/
        IDENTIFIER: /[A-Za-z]+/
        PARAM_NAME: /[a-zA-Z_]+/
        UNKNOWN_FUNC: /[a-zA-Z_]+/

        %import common.CNAME -> NAME
        %import common.NUMBER
        %import common.WS_INLINE
        %ignore WS_INLINE
    """

    kernel_parser = Lark(grammar, start="start")
    param_dict = {}
    kernel_transformer = CFGParamDictTransformer(param_dict)
    kernel_tree = kernel_parser.parse(str)
    # import pdb; pdb.set_trace()
    kernel = kernel_transformer.transform(kernel_tree)#.children[0]
    
    return kernel



if __name__=='__main__':
    example_with_params = 'LIN{offset:1} + PER{period:1, variance:1, lengthscale:1} * SE{variance:VAR_Y} * (PER{lengthscale:1} * LIN)'
    # example_without_params = "LIN + PER"
    example_without_params = "CP(LIN, LIN+PER*SE, x:199) + LIN"
    example_with_params2 = 'PER{variance: 1.1**2, period: 11, lengthscale: 1.0} * LIN{offset: 1.32**2}'
    example_with_params3 = "LIN{offset: 0.0145} * SE{variance: np.var(y), lengthscale: 0.0132}"
    example_with_params4 = "LIN{offset: 0.0145, offset: 0.02} * SE{variance: var(y), lengthscale: 0.0132}"
    
    
    # print('------parse_kernel_name test---------')
    # print(parse_kernel_name(example_with_params))
    # print(parse_kernel_name(example_with_params2))
    # print(parse_kernel_name(example_with_params3))
    # print(parse_kernel_name(example_with_params4))
    # print(parse_kernel_name(example_without_params))
    # # print('---------------------------------')
    # print('------parse_kernel_model test---------')
    # print(parse_kernel_model(example_with_params))
    # print(parse_kernel_model(example_with_params2))
    # print(parse_kernel_model(example_with_params3))
    # print(parse_kernel_model(example_with_params4))
    # print(parse_kernel_model(example_without_params))
    # print('---------------------------------')
    
    t = 'PER{variance: 0.0325, period: 0.7826, lengthscale: 0.0833} + SE{variance: 0.0325, lengthscale: 0.0833}'
    # print(parse_kernel_model(t))
    print(parse_kernel_params(t))
    print('------parse_kernel_model test---------')
    print(parse_kernel_params(example_with_params))
    print(parse_kernel_params(example_with_params2))
    print(parse_kernel_params(example_with_params3))
    print(parse_kernel_params(example_with_params4))
    print(parse_kernel_params(example_without_params))
    import pdb; pdb.set_trace()