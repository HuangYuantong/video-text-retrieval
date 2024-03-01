import torch
from torch import nn

A = torch.Tensor(3, 5, 10)
for i in range(5):
    for batch in range(3):
        A[batch, i, :] = batch * 100 + torch.tensor(list(range(i * 10, (i + 1) * 10)))

print(A)
print(A.norm(dim=1, keepdim=True))
print(A / A.norm(dim=1, keepdim=True))
exit()


def testLayerNorm():
    """LayerNorm"""
    for i in range(5):
        for batch in range(3):
            A[batch, i, :] = batch * 100 + torch.tensor(list(range(i * 10, (i + 1) * 10)))
    print(A)
    with torch.no_grad():
        ln = nn.LayerNorm(10)  # λ[10]=1, γ[10]=0
        # print(len(list(ln.parameters())), len(next(ln.parameters())), sep=', ')
        # # [3, 5, 10]
        print(ln(A))
        # # [5, 3, 10]
        # print(A.permute(1, 0, 2))
        # print(ln(A.permute(1, 0, 2)))

        print(ln(A[:, 0, :]))
        print(A.shape, A[:, 0, :].shape, ln(A[:, 0, :]).shape)


def testBatchNorm():
    """BatchNorm"""
    for i in range(5):
        for batch in range(3):
            temp = list(range(i * 10, (i + 1) * 10))
            # temp[-1] -= 9
            A[batch, i, :] = batch * 100 + torch.tensor(temp)
    print(A)
    with torch.no_grad():
        bn = nn.BatchNorm1d(5)  # λ[10]=1, γ[10]=0
        # print(len(list(bn.parameters())), len(next(bn.parameters())), sep=', ')
        print(bn(A))

    # B = torch.stack((A[0, 0, :], A[1, 0, :], A[2, 0, :]))
    # print(B)
    # B = (B - B.mean()) / B.std()
    # print(B)
