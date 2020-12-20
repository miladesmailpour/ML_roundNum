import Const
import Var
import PlaceHolder
import Opration
import Linear_Regression
import LossAndOptimizer


def const_1():  # Constant node and Session
    Const.simple_1('NOT used Session and Eager execution is enable!')
    Const.simple_2('Using Session and Eager execution is enable!')
    Const.simple_3('Using Session and Eager execution is disable!')
    Const.simple_4('Two Constant node printing out!!')


def var_1():  # Variable node, initializing and assign
    Var.var_declaration('Using Session and initialize, Eager execution is disable!')
    Var.var_assign('Using Session and initialize, Eager execution is disable!')


def placeholder_1():  # Place holder node
    PlaceHolder.single_placeholder('Print out single place holder!')
    PlaceHolder.multi_placeholder('Print out multi place holder!')


def operation_1():  # Operation node
    Opration.not_tf_operation_constant('NOT using Tensorflow Operation adding to constant nodes!')
    Opration.tf_operation_constant('Using Tensorflow Operation adding to constant nodes!')
    Opration.tf_operation_placeholder('Using Tensorflow Operation, adding constant and place holder node!')


def linear_regression_graph():  # making simple Linear Regression Graph "y = wx + b"
    Linear_Regression.lr_not_tf_operation('print out without tensorflow Operation')
    Linear_Regression.lr_tf_operation('print out Using tensorflow Operation')


def loss_optimizer():  # Loss function and Optimizer"
    LossAndOptimizer.simple_loss('print out without tensorflow Operation')


if __name__ == '__main__':
    # const_1()
    # var_1()
    # placeholder_1()
    # operation_1()
    # linear_regression_graph()
    loss_optimizer()
    print("\nDone!")

