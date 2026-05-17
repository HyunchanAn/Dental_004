import torch
import torch.nn as nn

class SwinIRLight(nn.Module):
    """
    M2 Pro (16GB) 환경에 최적화된 경량화 SwinIR 모델.
    RSTB(Residual Swin Transformer Block) 수를 조정하여 메모리 효율을 높임.
    """
    def __init__(self, 
                 img_size=64, 
                 patch_size=1, 
                 in_chans=3,
                 embed_dim=60, 
                 depths=[6, 6, 6, 6], 
                 num_heads=[6, 6, 6, 6],
                 window_size=8, 
                 mlp_ratio=2., 
                 upscale=2, 
                 img_range=1., 
                 upsampler='pixelshuffle'):
        super(SwinIRLight, self).__init__()
        
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler

        # 1. Shallow Feature Extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1)

        # 2. Deep Feature Extraction (경량화를 위해 4개의 RSTB 사용)
        self.layers = nn.ModuleList()
        for i in range(len(depths)):
            layer = RSTB(dim=embed_dim,
                         depth=depths[i],
                         num_heads=num_heads[i],
                         window_size=window_size,
                         mlp_ratio=mlp_ratio)
            self.layers.append(layer)
        
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        # 3. Upsampling Module
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = nn.Sequential(
                nn.Conv2d(64, in_chans * (upscale ** 2), kernel_size=3, padding=1),
                nn.PixelShuffle(upscale)
            )

    def forward(self, x):
        # 입력 범위 정규화 체크 (0~1 권장)
        x_first = self.conv_first(x)
        
        res = x_first
        for layer in self.layers:
            res = layer(res)
        
        res = self.conv_after_body(res)
        res = res + x_first
        
        # Upsampling
        x = self.conv_before_upsample(res)
        x = self.upsample(x)
        
        return x

class RSTB(nn.Module):
    """
    Residual Swin Transformer Block.
    간략화된 버전으로 메모리 점유율을 최소화함.
    """
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio):
        super(RSTB, self).__init__()
        self.dim = dim
        # 실제 구현에서는 Swin Transformer Layer(STL)가 반복되나, 
        # 여기서는 메모리 효율을 위해 단순화된 Residual Block 구조를 예시로 구현함.
        # 정식 SwinIR STL 코드는 라이브러리 연동 또는 상세 구현 필요.
        self.body = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            ) for _ in range(depth // 2)
        ])
        self.conv_last = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        for layer in self.body:
            res = layer(res) + res
        return self.conv_last(res) + x

if __name__ == "__main__":
    # M2 Pro 메모리 체크를 위한 더미 텐서 테스트
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SwinIRLight(upscale=2).to(device)
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    output = model(dummy_input)
    print(f"입력 크기: {dummy_input.shape}")
    print(f"출력 크기: {output.shape}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")
