import Const
import Var
import PlaceHolder
import Opration
import LinearRegressionGraph
import LossAndOptimizer
import Linear_Regression
import Estimator_Model
from LRM import Linear_Regression_Model as LRM_1


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
    LinearRegressionGraph.lr_not_tf_operation('print out without tensorflow Operation')
    LinearRegressionGraph.lr_tf_operation('print out Using tensorflow Operation')


def loss_optimizer():  # Loss function and Optimizer"
    # Not to excuse
    print("LossAndOptimizer.simple_loss('print out without tensorflow Operation')")


def simple_linear_regression_model():  # Linear Regression Model
    Linear_Regression.lr_model('Linear Regression Model:')


def estimator():  # Linear Regression Model via Estimator
    Estimator_Model.prebuilt_estimator('Linear Regression Model via Estimator:')
    Estimator_Model.custom_estimator('Linear Regression Model via custom Estimator:')


def final_model_lr():
    LRM_1.lr_model("creating 5 files include needed file .pbtxt and .ckpt")


if __name__ == '__main__':
    # const_1()
    # var_1()
    # placeholder_1()
    # operation_1()
    # linear_regression_graph()
    # loss_optimizer()
    # simple_linear_regression_model()
    # estimator()
    final_model_lr()
    print("\nDone!")

