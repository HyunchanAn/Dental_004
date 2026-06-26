import torch
import torch.nn as nn

class EdgeWeightedTVLoss(nn.Module):
    """
    Edge-Weighted Total Variation Loss (EW-TV).
    치조골이나 치아 경계 등 고주파 엣지 영역에서는 정규화 페널티를 완화하고,
    평탄한 연조직 영역에는 강한 TV 정규화를 부여하여 Staircase 현상을 방지합니다.
    """
    def __init__(self, edge_alpha=5.0):
        super(EdgeWeightedTVLoss, self).__init__()
        self.edge_alpha = edge_alpha

    def forward(self, sr, hr):
        """
        sr: 복원된 고해상도 이미지 텐서 (B, C, H, W)
        hr: 원본 고해상도 이미지 텐서 (B, C, H, W)
        """
        # 1. HR 원본의 그래디언트(엣지) 계산
        # 메모리 최적화를 위해 슬라이싱 사용 (GPU 병렬 처리에 유리)
        hr_diff_x = torch.abs(hr[:, :, :, 1:] - hr[:, :, :, :-1])
        hr_diff_y = torch.abs(hr[:, :, 1:, :] - hr[:, :, :-1, :])

        # 크기 맞추기 (H-1, W-1)
        hr_diff_x = hr_diff_x[:, :, :-1, :]
        hr_diff_y = hr_diff_y[:, :, :, :-1]
        
        # 엣지 맵 계산 (Magnitude)
        edge_map = torch.sqrt(hr_diff_x ** 2 + hr_diff_y ** 2 + 1e-8).detach()
        
        # 2. 가중치 매트릭스 G 계산 (에지가 강할수록 G는 0에 가까워짐)
        # G = exp(-alpha * E) 형태를 사용하여 평탄 영역(E~0)은 1, 엣지 영역(E>0)은 작아지게 함
        G = torch.exp(-self.edge_alpha * edge_map)

        # 3. SR 예측값의 그래디언트 계산
        sr_diff_x = torch.abs(sr[:, :, :-1, 1:] - sr[:, :, :-1, :-1])
        sr_diff_y = torch.abs(sr[:, :, 1:, :-1] - sr[:, :, :-1, :-1])

        # 4. 가중치가 적용된 TV Loss 계산
        ew_tv = G * (sr_diff_x + sr_diff_y)
        
        return ew_tv.mean()

class InverseProblemRegularizedLoss(nn.Module):
    """
    MREIT 논문(Lee et al.)의 Uniqueness 제약을 모방하기 위해
    기본 L1 손실과 Edge-Weighted TV 손실을 결합한 손실 함수입니다.
    """
    def __init__(self, lambda_tv=1e-4, edge_alpha=5.0):
        super(InverseProblemRegularizedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.ew_tv_loss = EdgeWeightedTVLoss(edge_alpha=edge_alpha)
        self.lambda_tv = lambda_tv

    def forward(self, sr, hr, current_lambda=None):
        """
        current_lambda: Warm-up 스케줄러를 통해 동적으로 전달받은 lambda_tv.
                        전달되지 않으면 초기 설정된 self.lambda_tv를 사용합니다.
        """
        l1 = self.l1_loss(sr, hr)
        
        # lambda가 0인 경우 연산 절약을 위해 TV 연산 생략
        weight = current_lambda if current_lambda is not None else self.lambda_tv
        if weight <= 0.0:
            return l1, l1.item(), 0.0

        tv = self.ew_tv_loss(sr, hr)
        total_loss = l1 + weight * tv
        
        return total_loss, l1.item(), tv.item()
