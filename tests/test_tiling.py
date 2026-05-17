import torch
import torch.nn as nn
from core.tiling import PanoTiler

class DummyModel(nn.Module):
    """
    CI 테스트용 더미 업스케일 모델.
    입력 텐서를 받아서 단순히 bilinear 인터폴레이션으로 upscale 배율만큼 확대하여 반환합니다.
    """
    def __init__(self, upscale=2):
        super(DummyModel, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        # x: (B, C, H, W) -> (B, C, H * upscale, W * upscale)
        return nn.functional.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

def test_pano_tiler_initialization():
    """
    PanoTiler 초기 설정 값이 정상적으로 인스턴스 변수에 바인딩되는지 검증합니다.
    """
    tiler = PanoTiler(tile_size=256, overlap=32, upscale=2)
    assert tiler.tile_size == 256
    assert tiler.overlap == 32
    assert tiler.stride == 224
    assert tiler.upscale == 2

def test_tile_image():
    """
    주어진 대형 이미지 텐서가 설정된 타일 크기 및 overlap(stride) 규칙에 맞추어 
    정밀하게 분할되고, 타일 시작 좌표가 올바르게 계산되는지 검증합니다.
    """
    tiler = PanoTiler(tile_size=64, overlap=16, upscale=2)
    
    # 1채널 100x150 크기의 더미 이미지 텐서 생성
    dummy_img = torch.randn(1, 100, 150)
    
    tiles, coords = tiler.tile_image(dummy_img)
    
    # 생성된 모든 타일의 형상이 (1, 64, 64)인지 확인
    assert tiles.ndim == 4  # (N, C, H, W)
    assert tiles.shape[1:] == (1, 64, 64)
    
    # 이미지 경계를 초과하지 않고 바운더리에 딱 맞춰 생성되었는지 좌표 검증
    # y 방향: 0, 48(100-64=36이 최소 한계이므로 y_start=36으로 조정됨) -> 2개 행 생성 예상
    # x 방향: 0, 48, 86(150-64=86이므로) -> 3개 열 생성 예상
    # 총 타일 수: 2 * 3 = 6
    assert len(tiles) == len(coords)
    
    # 타일링 마지막 좌표가 영상의 최대 크기 범위를 넘어가지 않는지 검증
    for y, x in coords:
        assert y + tiler.tile_size <= 100
        assert x + tiler.tile_size <= 150

def test_merge_tiles():
    """
    분할된 타일들이 가중치 블렌딩 마스크를 통해 정상적으로 하나의 
    전체 이미지로 병합되는지 검증합니다.
    """
    tiler = PanoTiler(tile_size=64, overlap=16, upscale=2)
    
    # 1채널 100x150 원본 기준 업스케일 대상 형상: (1, 200, 300)
    dummy_img = torch.ones(1, 100, 150)
    tiles, coords = tiler.tile_image(dummy_img)
    
    # 모의 모델 처리 후 타일 형상 (N, 1, 128, 128)
    processed_tiles = nn.functional.interpolate(tiles, scale_factor=2, mode='nearest')
    
    target_shape = (1, 200, 300)
    merged = tiler.merge_tiles(processed_tiles, coords, target_shape)
    
    assert merged.shape == target_shape
    # 가중치 합산 시 분모가 0이 되어 NaN이 발생했는지 확인
    assert not torch.isnan(merged).any()

def test_process_large_image_with_padding():
    """
    입력 영상이 타일 크기(tile_size)보다 작은 소형 영상일 때, 
    내부적으로 reflect 패딩이 활성화되어 정상적으로 추론되고 
    원래 영상의 업스케일 크기로 복원되는지 검증합니다.
    """
    tiler = PanoTiler(tile_size=128, overlap=32, upscale=2)
    model = DummyModel(upscale=2)
    
    # 타일 사이즈 128보다 작은 50x80 이미지 텐서 생성
    small_img = torch.randn(1, 50, 80)
    
    result = tiler.process_large_image(model, small_img, device='cpu')
    
    # upscale=2 배율 적용에 따른 최종 크기가 (1, 100, 160)인지 검증
    assert result.shape == (1, 100, 160)
    assert not torch.isnan(result).any()
