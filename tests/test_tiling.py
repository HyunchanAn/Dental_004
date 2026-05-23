import torch
import torch.nn as nn
from pano_clear.tiling import PanoTiler

class DummyModel(nn.Module):
    """
    CI ?뚯뒪?몄슜 ?붾? ?낆뒪耳??紐⑤뜽.
    ?낅젰 ?먯꽌瑜?諛쏆븘???⑥닚??bilinear ?명꽣?대젅?댁뀡?쇰줈 upscale 諛곗쑉留뚰겮 ?뺣??섏뿬 諛섑솚?⑸땲??
    """
    def __init__(self, upscale=2):
        super(DummyModel, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        # x: (B, C, H, W) -> (B, C, H * upscale, W * upscale)
        return nn.functional.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

def test_pano_tiler_initialization():
    """
    PanoTiler 珥덇린 ?ㅼ젙 媛믪씠 ?뺤긽?곸쑝濡??몄뒪?댁뒪 蹂?섏뿉 諛붿씤?⑸릺?붿? 寃利앺빀?덈떎.
    """
    tiler = PanoTiler(tile_size=256, overlap=32, upscale=2)
    assert tiler.tile_size == 256
    assert tiler.overlap == 32
    assert tiler.stride == 224
    assert tiler.upscale == 2

def test_tile_image():
    """
    二쇱뼱吏?????대?吏 ?먯꽌媛 ?ㅼ젙??????ш린 諛?overlap(stride) 洹쒖튃??留욎텛??
    ?뺣??섍쾶 遺꾪븷?섍퀬, ????쒖옉 醫뚰몴媛 ?щ컮瑜닿쾶 怨꾩궛?섎뒗吏 寃利앺빀?덈떎.
    """
    tiler = PanoTiler(tile_size=64, overlap=16, upscale=2)
    
    # 1梨꾨꼸 100x150 ?ш린???붾? ?대?吏 ?먯꽌 ?앹꽦
    dummy_img = torch.randn(1, 100, 150)
    
    tiles, coords = tiler.tile_image(dummy_img)
    
    # ?앹꽦??紐⑤뱺 ??쇱쓽 ?뺤긽??(1, 64, 64)?몄? ?뺤씤
    assert tiles.ndim == 4  # (N, C, H, W)
    assert tiles.shape[1:] == (1, 64, 64)
    
    # ?대?吏 寃쎄퀎瑜?珥덇낵?섏? ?딄퀬 諛붿슫?붾━????留욎떠 ?앹꽦?섏뿀?붿? 醫뚰몴 寃利?
    # y 諛⑺뼢: 0, 48(100-64=36??理쒖냼 ?쒓퀎?대?濡?y_start=36?쇰줈 議곗젙?? -> 2媛????앹꽦 ?덉긽
    # x 諛⑺뼢: 0, 48, 86(150-64=86?대?濡? -> 3媛????앹꽦 ?덉긽
    # 珥?????? 2 * 3 = 6
    assert len(tiles) == len(coords)
    
    # ??쇰쭅 留덉?留?醫뚰몴媛 ?곸긽??理쒕? ?ш린 踰붿쐞瑜??섏뼱媛吏 ?딅뒗吏 寃利?
    for y, x in coords:
        assert y + tiler.tile_size <= 100
        assert x + tiler.tile_size <= 150

def test_merge_tiles():
    """
    遺꾪븷????쇰뱾??媛以묒튂 釉붾젋??留덉뒪?щ? ?듯빐 ?뺤긽?곸쑝濡??섎굹??
    ?꾩껜 ?대?吏濡?蹂묓빀?섎뒗吏 寃利앺빀?덈떎.
    """
    tiler = PanoTiler(tile_size=64, overlap=16, upscale=2)
    
    # 1梨꾨꼸 100x150 ?먮낯 湲곗? ?낆뒪耳??????뺤긽: (1, 200, 300)
    dummy_img = torch.ones(1, 100, 150)
    tiles, coords = tiler.tile_image(dummy_img)
    
    # 紐⑥쓽 紐⑤뜽 泥섎━ ??????뺤긽 (N, 1, 128, 128)
    processed_tiles = nn.functional.interpolate(tiles, scale_factor=2, mode='nearest')
    
    target_shape = (1, 200, 300)
    merged = tiler.merge_tiles(processed_tiles, coords, target_shape)
    
    assert merged.shape == target_shape
    # 媛以묒튂 ?⑹궛 ??遺꾨え媛 0???섏뼱 NaN??諛쒖깮?덈뒗吏 ?뺤씤
    assert not torch.isnan(merged).any()

def test_process_large_image_with_padding():
    """
    ?낅젰 ?곸긽??????ш린(tile_size)蹂대떎 ?묒? ?뚰삎 ?곸긽???? 
    ?대??곸쑝濡?reflect ?⑤뵫???쒖꽦?붾릺???뺤긽?곸쑝濡?異붾줎?섍퀬 
    ?먮옒 ?곸긽???낆뒪耳???ш린濡?蹂듭썝?섎뒗吏 寃利앺빀?덈떎.
    """
    tiler = PanoTiler(tile_size=128, overlap=32, upscale=2)
    model = DummyModel(upscale=2)
    
    # ????ъ씠利?128蹂대떎 ?묒? 50x80 ?대?吏 ?먯꽌 ?앹꽦
    small_img = torch.randn(1, 50, 80)
    
    result = tiler.process_large_image(model, small_img, device='cpu')
    
    # upscale=2 諛곗쑉 ?곸슜???곕Ⅸ 理쒖쥌 ?ш린媛 (1, 100, 160)?몄? 寃利?
    assert result.shape == (1, 100, 160)
    assert not torch.isnan(result).any()
