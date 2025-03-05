from src.model.logit import RILogit

# TODO specify state of the world, a value for N, or a vector that contains each state of the world for each products, then use the conditionnal logit probabilities for that state of the world, use inverse sampling method , 1000
# Multiple individuals, (corresponds to demographic groups)


class SimulateRILogit(RILogit):
    def __init__(self, u_mat, ppi, llambda, method="BA", stop_fun="DIE", **kwargs):
        super().__init__(u_mat, ppi, llambda, method, stop_fun, **kwargs)
