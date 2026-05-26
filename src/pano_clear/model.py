import torch
import torch.nn as nn

class SwinIRLight(nn.Module):
    """
    M2 Pro (16GB) ?섍꼍??理쒖쟻?붾맂 寃쎈웾??SwinIR 紐⑤뜽.
    RSTB(Residual Swin Transformer Block) ?섎? 議곗젙?섏뿬 硫붾え由??⑥쑉???믪엫.
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

        # 2. Deep Feature Extraction (寃쎈웾?붾? ?꾪빐 4媛쒖쓽 RSTB ?ъ슜)
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
        # ?낅젰 踰붿쐞 ?뺢퇋??泥댄겕 (0~1 沅뚯옣)
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
    媛꾨왂?붾맂 踰꾩쟾?쇰줈 硫붾え由??먯쑀?⑥쓣 理쒖냼?뷀븿.
    """
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio):
        super(RSTB, self).__init__()
        self.dim = dim
        # ?ㅼ젣 援ы쁽?먯꽌??Swin Transformer Layer(STL)媛 諛섎났?섎굹, 
        # ?ш린?쒕뒗 硫붾え由??⑥쑉???꾪빐 ?⑥닚?붾맂 Residual Block 援ъ“瑜??덉떆濡?援ы쁽??
        # ?뺤떇 SwinIR STL 肄붾뱶???쇱씠釉뚮윭由??곕룞 ?먮뒗 ?곸꽭 援ы쁽 ?꾩슂.
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
    # 크로스 플랫폼 디바이스 자동 감지 (Issue #1)
    from pano_clear.device import get_best_device
    device = get_best_device()
    model = SwinIRLight(upscale=2).to(device)
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    output = model(dummy_input)
    print(f"사용 디바이스: {device}")
    print(f"입력 크기: {dummy_input.shape}")
    print(f"출력 크기: {output.shape}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")
