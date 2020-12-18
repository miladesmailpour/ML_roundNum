import Const
import Var


def const_1():  # Constant node and Session
    Const.simple_1('NOT used Session and Eager execution is enable!')
    Const.simple_2('Using Session and Eager execution is enable!')
    Const.simple_3('Using Session and Eager execution is disable!')
    Const.simple_4('Two Constant node printing out!!')


def var_1():  # Variable node, initializing and assign
    Var.var_declaration('Using Session and initialize, Eager execution is disable!')
    Var.var_assign('Using Session and initialize, Eager execution is disable!')


if __name__ == '__main__':
    # const_1()
    # var_1()
    print("\nDone!")

