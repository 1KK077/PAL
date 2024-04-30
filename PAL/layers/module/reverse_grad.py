from torch import nn
from torch.autograd import Function


class ReverseGradFunction(Function):

    @staticmethod
    def forward(ctx, data, alpha=1.0):
        ctx.alpha = alpha
        return data

    @staticmethod
    def backward(ctx, grad_outputs):
        grad = None

        if ctx.needs_input_grad[0]:
            # import pdb
            # pdb.set_trace()
            grad = -ctx.alpha * grad_outputs

        return grad, None


class ReverseGrad(nn.Module):
    def __init__(self, alpha):
        super(ReverseGrad, self).__init__()
        self.alpha = alpha
    def forward(self, x):
        alpha = self.alpha
        return ReverseGradFunction.apply(x, alpha)
