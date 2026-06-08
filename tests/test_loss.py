import torch
import torch.nn as nn
from pano_clear.loss import InverseProblemRegularizedLoss
from pano_clear.device import get_best_device

def test_tv_loss_device_and_shape():
    device = get_best_device()
    pred = torch.rand(2, 1, 256, 256, device=device, requires_grad=True)
    gt = torch.rand(2, 1, 256, 256, device=device)
    
    criterion = InverseProblemRegularizedLoss(lambda_tv=0.01)
    # 람다가 0보다 커야 TV 연산을 수행함
    loss, l1_val, tv_val = criterion(pred, gt, current_lambda=0.01)
    
    assert not torch.isnan(loss), "Loss contains NaN values"
    assert loss.device.type == device.type, "Device mismatch in loss computation"
    
def test_loss_backward_compatibility():
    device = get_best_device()
    
    # 가벼운 서브 네트워크 선언
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 1, 3, padding=1)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = InverseProblemRegularizedLoss(lambda_tv=0.01)
    
    dummy_input = torch.rand(2, 1, 64, 64, device=device)
    dummy_gt = torch.rand(2, 1, 64, 64, device=device)
    
    # 추론 및 역전파 흐름 테스트
    optimizer.zero_grad()
    out = model(dummy_input)
    loss, l1_val, tv_val = criterion(out, dummy_gt, current_lambda=0.01)
    loss.backward()
    
    # 모든 파라미터에 그래디언트가 정상 주입되었는지 확인
    for param in model.parameters():
        assert param.grad is not None, "Gradient graph is broken by TV Loss"
