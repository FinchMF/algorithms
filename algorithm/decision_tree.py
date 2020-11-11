
import time
import numpy as np 

class Decision_Tree:

    def __init__(self, depth=5, min_leaf_size=5, verbose=False):

        self.depth = depth
        self.decision_boundary = 0
        self.left = None
        self.right = None
        self.min_leaf_size = min_leaf_size
        self.preds = None

        self.verbose = verbose

    def mean_squared_error(self, labels, preds):

        if labels.ndim != 1:

            print(f"Error: Expected Label Input {1} but got {labels.ndim}")

        return np.mean((labels - preds) ** 2)

    def train(self, X, y):

        if X.ndim != 1:

            print(f"Error: Expected Data Input {1} but got {X.ndim}")
            return

        if len(X) != len(y):

            print(f"Format Error: X = {len(X)} and y = {len(y)}")
            return
        if y.ndim != 1:

            print(f"Error: Expected Label Input -- Expected Input {1} but got {y.ndim}")
            return

        if len(X) < 2 * self.min_leaf_size:

            self.preds = np.mean(y)
            return

        if self.depth == 1:

            self.preds = np.mean(y)
            return

        best_split = 0
        min_error = self.mean_squared_error(X,np.mean(y)) * 2

        start = time.time()
        for i in range(len(X)):
 
            if len(X[:i]) < self.min_leaf_size:

                continue
            elif len(X[i:]) < self.min_leaf_size:

                continue
            else:

                error_left = self.mean_squared_error(X[:i], np.mean(y[:i]))
                error_right = self.mean_squared_error(X[i:], np.mean(y[i:]))

                error = error_left + error_right

                if error < min_error:

                    best_split = i
                    min_error = error
        end = time.time()

        if self.verbose:
            print(f' ------ Inner loop Time Interval: {end-start} seconds ----- STEP: {i}')

        if best_split != 0:
            
            start = time.time()

            left_X = X[:best_split]
            left_y = y[:best_split]

            right_X = X[best_split:]
            right_y = y[best_split:]

            self.decision_boundary = X[best_split]

            self.left = Decision_Tree(depth = self.depth - 1, min_leaf_size = self.min_leaf_size)
            self.right = Decision_Tree(depth = self.depth - 1, min_leaf_size = self.min_leaf_size)

            self.left.train(left_X, left_y)
            self.right.train(right_X, right_y)
            end = time.time()

            if self.verbose:
                print(f' --------- Best Split Loop Time Interval: {end-start} seconds ------- STEP: {i}')
            else:
                pass 
        else:

            self.preds = np.mean(y)
            end = time.time()

            if self.verbose:
                print(f' --------- Best Split Loop Time Interval: {end-start} seconds --------- STEP: {i}')
            else:
                pass
        return

    def predict(self, x):

        if self.preds is not None:

            return self.preds
        elif self.left or self.right is not None:

            if x >= self.decision_boundary:

                return self.right.predict(x)
            else:

                return self.left.predict(x)
        else:
            print("Error: Decision tree did not sufficently train")
            return None


class Evaluate:

    def __init__(self, decision_tree, output_size=10):

        self.X = np.arange(-1., 1., 0.005)
        self.y = np.sin(self.X)

        self.tree = decision_tree
        self.output_size = output_size

    def asess(self):

        start = time.time()
        test_cases = (np.random.rand(self.output_size) * 2) - 1
        self.tree.train(self.X, self.y)
        preds = np.array([self.tree.predict(x) for x in test_cases])
        avg_err = np.mean((preds - test_cases) ** 2)
        end = time.time()

        print(f' --- Time: {end-start} seconds\n')
        print(f' || Test Values: \
                {test_cases}\n')
        print(f' || Preds: \
                {preds}\n')
        print(f' || Average err: \
                {avg_err}\n')

        return 


if __name__ == '__main__':

    tree = Decision_Tree(depth=10, min_leaf_size=10)
    Evaluate(tree).asess()





