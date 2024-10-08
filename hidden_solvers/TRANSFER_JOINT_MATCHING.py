from benchopt import safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    from skada import TransferJointMatchingAdapter, make_da_pipeline
    from benchmark_utils.base_solver import DASolver, FinalEstimator


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(DASolver):
    # Name to select the solver in the CLI and to display the results.
    name = 'transfer_joint_matching'

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    default_param_grid = {
        'transferjointmatchingadapter__kernel': ['rbf'],
        'transferjointmatchingadapter__n_components': [1, 2, 5, 10, 20, 50, 100],
        'transferjointmatchingadapter__tradeoff': [0, 1e-2],
        'finalestimator__estimator_name': ["LR", "SVC", "XGB"],
    }

    def get_estimator(self):
        # return CORAL()
        # The estimator passed should have a 'predict_proba' method.
        return make_da_pipeline(
            TransferJointMatchingAdapter(),
            FinalEstimator(),
        )
