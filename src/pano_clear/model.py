import torch
import torch.nn as nn

class SwinIRLight(nn.Module):
    """
    M2 Pro (16GB) ?˜ê²½??ìµœì ?”ëœ ê²½ëŸ‰??SwinIR ëª¨ë¸.
    RSTB(Residual Swin Transformer Block) ?˜ë? ì¡°ì •?˜ì—¬ ë©”ëª¨ë¦??¨ìœ¨???’ì„.
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

        # 2. Deep Feature Extraction (ê²½ëŸ‰?”ë? ?„í•´ 4ê°œì˜ RSTB ?¬ìš©)
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
        # ?…ë ¥ ë²”ìœ„ ?•ê·œ??ì²´í¬ (0~1 ê¶Œì¥)
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
    ê°„ëµ?”ëœ ë²„ì „?¼ë¡œ ë©”ëª¨ë¦??ìœ ?¨ì„ ìµœì†Œ?”í•¨.
    """
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio):
        super(RSTB, self).__init__()
        self.dim = dim
        # ?¤ì œ êµ¬í˜„?ì„œ??Swin Transformer Layer(STL)ê°€ ë°˜ë³µ?˜ë‚˜, 
        # ?¬ê¸°?œëŠ” ë©”ëª¨ë¦??¨ìœ¨???„í•´ ?¨ìˆœ?”ëœ Residual Block êµ¬ì¡°ë¥??ˆì‹œë¡?êµ¬í˜„??
        # ?•ì‹ SwinIR STL ì½”ë“œ???¼ì´ë¸ŒëŸ¬ë¦??°ë™ ?ëŠ” ?ì„¸ êµ¬í˜„ ?„ìš”.
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
    # M2 Pro ë©”ëª¨ë¦?ì²´í¬ë¥??„í•œ ?”ë? ?ì„œ ?ŒìŠ¤??
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SwinIRLight(upscale=2).to(device)
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    output = model(dummy_input)
    print(f"?…ë ¥ ?¬ê¸°: {dummy_input.shape}")
    print(f"ì¶œë ¥ ?¬ê¸°: {output.shape}")
    print(f"ëª¨ë¸ ?Œë¼ë¯¸í„° ?? {sum(p.numel() for p in model.parameters())}")
