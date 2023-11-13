import torch


class Indexer(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        x,i = inputs
        i = i * 65
        if i < 0:
            i = 0
        if i > 65:
            i = 65
        down = int(i.item())
        up = int(i.item())+1
        y_down = x[:,down]
        y_up = x[:,up]
        slope = y_down-y_up
        ctx.save_for_backward(slope)

    @staticmethod
    def forward(x, i):
        i = i * 65
        if i < 0:
            i = 0
        if i > 65:
            i = 65
        down = int(i.item())
        up = int(i.item())+1
        x_diff = i.item()-down
        y_down = x[:,down]
        y_up = x[:,up]
        slope = y_down-y_up
        y = slope*x_diff
        return y

    @staticmethod
    def backward(ctx, grad_output):
        slope = ctx.saved_tensors[0]
        return None, slope*grad_output


if __name__ == "__main__":
    x = torch.tensor([[10, 9, 8, 7, 6, 5]])
    i = torch.tensor([2.2], requires_grad=True)

    custom_module = Indexer.apply
    y = custom_module(x, i)
    y.backward()

    grad_i = i.grad

    print("Gradient of i:", grad_i.item())
