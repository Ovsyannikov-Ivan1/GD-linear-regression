import numpy as np
from scipy.sparse.linalg import svds
from interfaces import LossFunction, LossFunctionClosedFormMixin, LinearRegressionInterface, AbstractOptimizer
from descents import AnalyticSolutionOptimizer
from typing import Dict, Type, Optional, Callable
from abc import abstractmethod, ABC



class MSELoss(LossFunction, LossFunctionClosedFormMixin):

    def __init__(self, analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):

        if analytic_solution_func is None:
            self.analytic_solution_func = self._plain_analytic_solution
        else:
            self.analytic_solution_func = analytic_solution_func

        

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: float, значение MSE на данных X,y для весов w
        """
        return np.mean( (X @ w - y) **2 )

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета
        w: np.ndarray, вектор весов

        returns: np.ndarray, численный градиент MSE в точке w
        """
        n = X.shape[0]
        return 2 * X.T @ (X @ w - y) / X.shape[0]

    def analytic_solution(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Возвращает решение по явной формуле (closed-form solution)

        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, оптимальный по MSE вектор весов, вычисленный при помощи аналитического решения для данных X, y
        """
        # Функция-диспатчер в одну из истинных функций для вычисления решения по явной формуле (closed-form)
        # Необходима в связи c наличием интерфейса analytic_solution у любого лосса; 
        # self-injection даёт возможность выбирать, какое именно closed-form решение использовать
        return self.analytic_solution_func(X, y)
        
    
    @classmethod
    def _plain_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи классического аналитического решения
        """
        return np.linalg.pinv(X.T @ X) @ X.T @ y
    
    @classmethod
    def _svd_analytic_solution(cls, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, матрица регрессоров 
        y: np.ndarray, вектор таргета

        returns: np.ndarray, вектор весов, вычисленный при помощи аналитического решения на SVD
        """
        n, m = X.shape    
        k_svd = max(1, min(n, m) - 1)
        U, sigma, VT = svds(X, k_svd)
        
        return VT.T @ ((U.T @ y) / sigma)


class L2Regularization(LossFunction):

    def __init__(self, core_loss: LossFunction, mu_rate: float = 1.0,
                 analytic_solution_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        
        self.core_loss = core_loss
        self.mu_rate = mu_rate

        # analytic_solution_func is meant to be passed separately, 
        # as it is not linear to core solution
        
    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        return self.core_loss.loss(X, y, w) + self.mu_rate * (w @ w)
    

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:

        core_part = self.core_loss.gradient(X, y, w)

        penalty_part = core_part + self.mu_rate * w
        return penalty_part



class CustomLinearRegression(LinearRegressionInterface):
    def __init__(
        self,
        optimizer: AbstractOptimizer,
        # l2_coef: float = 0.0,
        loss_function: LossFunction = MSELoss()
    ):
        self.optimizer = optimizer
        self.optimizer.set_model(self)

        # self.l2_coef = l2_coef
        self.loss_function = loss_function
        self.loss_history = []
        self.w = None
        self.X_train = None
        self.y_train = None
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        returns: np.ndarray, вектор \hat{y}
        """
        return X @ self.w

    def compute_gradients(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> np.ndarray:
        """
        returns: np.ndarray, градиент функции потерь при текущих весах (self.w)
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if X_batch is None:
            X_batch = self.X_train
        if y_batch is None:
            y_batch = self.y_train
        return self.loss_function.gradient(X_batch, y_batch, self.w)


    def compute_loss(self, X_batch: np.ndarray | None = None, y_batch: np.ndarray | None = None) -> float:
        """
        returns: np.ndarray, значение функции потерь при текущих весах (self.w) по self.X_train, self.y_train
        Если переданы аргументы, то градиент вычисляется по ним, иначе - по self.X_train и self.y_train
        """
        if X_batch is None:
            X_batch = self.X_train
        if y_batch is None:
            y_batch = self.y_train
        return self.loss_function.loss(X_batch, y_batch, self.w)


    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Инициирует обучение модели заданным функцией потерь и оптимизатором способом.
        
        X: np.ndarray, 
        y: np.ndarray
        """
        self.X_train, self.y_train = X, y
        self.optimizer.optimize()




class LogCoshLoss(LossFunction):

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        z = X @ w - y
        a = np.abs(z)
        return np.mean(a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0))

    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        z = X @ w - y
        return (X.T @ np.tanh(z)) / X.shape[0]


class HuberLoss(LossFunction):

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def loss(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        q = 0.5 * (X @ w - y) ** 2
        line = self.delta * (np.abs(X @ w - y)) - 0.5 * self.delta ** 2
        return float(np.mean(np.where((np.abs(X @ w - y)) < self.delta, q, line)))


    def gradient(self, X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        h = np.clip(X @ w - y, -self.delta, self.delta)
        return (1 / len(y)) * X.T @ h
