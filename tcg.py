import torch
from functools import reduce
from torch.optim import Optimizer

class TCG(Optimizer):
    """Implements TCG algorithm, inspired by L-BFGS
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    Arguments:
        delta (float): trust region size
        kmax (int): maximal number of iterations per optimization step
        tol (float): termination tolerance
        eps (float): termination tolerance
    """

    def __init__(self, params, delta, kmax, tol, eps, momentum):

        defaults = dict(delta=delta,
                        kmax=kmax,
                        tol=tol,
                        eps=eps,
                        momentum=momentum)
        super(TCG, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("TCG doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def flat_grad(self, grads):
        views = []
        for g in grads:
            if g is None:
                print("Here be dragons")
                view = g.new(g.numel()).zero_()
            elif g.is_sparse:
                print("Warning sparse grad")
                view = g.to_dense().view(-1)
            else:
                view = g.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset:offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    @torch.no_grad()
    def step(self, loss):
        """Performs a single optimization step.

        Arguments:
            loss : the loss function result
                   in order to compute grad and Hess here
        Returns:
            flag, k, qval :
            Model parameters are updated
        """
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        delta = group["delta"]
        kmax = group["kmax"]
        tol = group["tol"]
        eps = group["eps"]
        momentum = group["momentum"]

        state = self.state[self._params[0]]
        prev_grad = state.get('d')

        # Computes the grad, enable_grad to compute the graph
        # for Hess computation
        with torch.enable_grad():
            grads = torch.autograd.grad(loss,
                                        self._params,
                                        create_graph=True,
                                        retain_graph=True)
            grad = self.flat_grad(grads)

        g, g0 = torch.clone(grad), torch.clone(grad)
        if prev_grad is not None:
            g += momentum*prev_grad
        ng0 = torch.norm(g0)
        s = torch.zeros_like(g, requires_grad=False)
        p = g.neg()

        noconvergence = False
        if ng0>eps:
            noconvergence = True
        else:
            print("Gradient %.4f - TCG"%(ng0.item()))

        k, ng, flag, qval = 0, ng0, 0, 0
        while noconvergence:
            k += 1

            # Computes Hess times p
            Hps = torch.autograd.grad(grad,
                                      self._params,
                                      grad_outputs=p,
                                      retain_graph=True)
            Hp = self.flat_grad(Hps)

            kappa = torch.dot(p, Hp)
            if kappa<=0:
                tmp1 = torch.dot(p, s)
                tmp2 = torch.dot(p, p)
                tmp3 = torch.dot(s, s)
                tmp4 = torch.dot(g, p)
                qval = -0.5*torch.dot(s, g0)
                disc = tmp1**2 + tmp2*(delta**2 - tmp3)
                sigma = (disc**0.5 - tmp1)/tmp2
                s += sigma*p
                flag = -1
                qval += -sigma*(tmp4 + 0.5*sigma*kappa)

                noconvergence = False
            else:
                alpha = ng**2/kappa
                sp1 = s + alpha*p

                if torch.dot(sp1, sp1) > delta**2:
                    tmp1 = torch.dot(p, s)
                    tmp2 = torch.dot(p, p)
                    tmp3 = torch.dot(s, s)
                    tmp4 = torch.dot(g, p)
                    qval = -0.5*torch.dot(s, g0)
                    disc = tmp1**2 + tmp2*(delta**2 - tmp3)
                    sigma = (disc**0.5 - tmp1)/tmp2
                    s += sigma*p
                    flag = -2
                    qval += -sigma*(tmp4 + 0.5*sigma*kappa)

                    noconvergence = False
                else:

                    s = sp1
                    g += alpha*Hp
                    ngp1sq = torch.dot(g, g)
                    p = -g + p*ngp1sq/ng**2
                    ng = ngp1sq**0.5

            if k >= kmax:
                noconvergence = False
                flag = -3
                qval = -0.5*torch.dot(s, g0)

        self._add_grad(1., s) # update model parameters
        state['d'] = g0
        return flag, k, qval
