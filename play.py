import torch


class CustomLinearModule(torch.autograd.Function):
    @staticmethod
    def setup_context(ctx, inputs, output):
        x,i = inputs
        ctx.save_for_backward(x,i)

    @staticmethod
    def forward(x, i):
        low = int(i.item())
        up = int(i.item())+1
        diff = i.item()-low
        low = x[low]
        up = x[up]
        return (up-low)*torch.tensor(diff)

    @staticmethod
    def backward(ctx, grad_output):
        x, i = ctx.saved_tensors
        low = int(i.item())
        up = int(i.item())+1
        diff = i.item()-low
        low = x[low]
        up = x[up]
        return None,(up-low).reshape(-1,1)


x = torch.tensor([10,9,8,7,6,5])
i = torch.tensor([2.2], requires_grad=True)

custom_module = CustomLinearModule.apply
y = custom_module(x, i)
y.backward()

grad_i = i.grad

print("Gradient of i:", grad_i.item())
