from src.model.logit import RILogit


class SimulateRILogit(RILogit):
    def __init__(self, u_mat, ppi, llambda, method="BA", stop_fun="DIE", **kwargs):
        super().__init__(u_mat, ppi, llambda, method, stop_fun, **kwargs)
