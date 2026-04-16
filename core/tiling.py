import torch
import torch.nn.functional as F
import numpy as np
import cv2

class PanoTiler:
    """
    고해상도 파노라마 이미지를 타일 단위로 분할하고 합치는 클래스.
    경계면의 부자연스러움을 방지하기 위해 Overlapping 및 Weighted Blending 적용.
    """
    def __init__(self, tile_size=512, overlap=64, upscale=2):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.upscale = upscale

    def tile_image(self, img):
        """
        이미지를 중첩되는 타일로 분할.
        img: (C, H, W) Tensor
        """
        c, h, w = img.shape
        tiles = []
        coords = []

        for y in range(0, h - self.overlap, self.stride):
            for x in range(0, w - self.overlap, self.stride):
                # 이미지 경계를 벗어나지 않도록 조정
                y_start = min(y, h - self.tile_size)
                x_start = min(x, w - self.tile_size)
                
                tile = img[:, y_start:y_start+self.tile_size, x_start:x_start+self.tile_size]
                tiles.append(tile)
                coords.append((y_start, x_start))
                
                if x_start + self.tile_size >= w: break
            if y_start + self.tile_size >= h: break

        return torch.stack(tiles), coords

    def merge_tiles(self, tiles, coords, target_shape):
        """
        타일들을 합쳐서 전체 이미지 복원.
        tiles: (N, C, T, T) Tensor (Processed tiles)
        coords: 타일의 시작 좌표 리스트 (원본 기준)
        target_shape: (C, H_large, W_large)
        """
        c, h_large, w_large = target_shape
        output = torch.zeros(target_shape, device=tiles.device)
        weights = torch.zeros(target_shape, device=tiles.device)

        # 타일 블렌딩을 위한 윈도우 마스크 생성 (경계면을 부드럽게)
        mask = self._get_mask(self.tile_size * self.upscale).to(tiles.device)

        for i, (y, x) in enumerate(coords):
            y_up, x_up = y * self.upscale, x * self.upscale
            t_size_up = self.tile_size * self.upscale
            
            # 타일이 이미지 경계를 벗어날 경우를 대비한 슬라이싱 계산
            y_end = min(y_up + t_size_up, h_large)
            x_end = min(x_up + t_size_up, w_large)
            
            tile_h = y_end - y_up
            tile_w = x_end - x_up
            
            # 타일과 마스크를 경계 크기에 맞게 자름 (Clips if necessary)
            curr_tile = tiles[i][:, :tile_h, :tile_w]
            curr_mask = mask[:tile_h, :tile_w]
            
            output[:, y_up:y_end, x_up:x_end] += curr_tile * curr_mask
            weights[:, y_up:y_end, x_up:x_end] += curr_mask

        return output / (weights + 1e-8)

    def _get_mask(self, size):
        """
        타일 테두리로 갈수록 투명해지는 선형 코사인 마스크 생성.
        """
        mask_1d = torch.linspace(0, 1, steps=self.overlap * self.upscale)
        mask_1d = 0.5 - 0.5 * torch.cos(mask_1d * 3.14159)
        
        full_mask = torch.ones((size, size))
        transition = self.overlap * self.upscale
        
        # 상하좌우 경계 처리
        full_mask[:transition, :] *= mask_1d.view(-1, 1)
        full_mask[-transition:, :] *= mask_1d.flip(0).view(-1, 1)
        full_mask[:, :transition] *= mask_1d.view(1, -1)
        full_mask[:, -transition:] *= mask_1d.flip(0).view(1, -1)
        
        return full_mask

    def process_large_image(self, model, img_tensor, device):
        """
        모델을 사용하여 대용량 이미지 전체를 타일링 방식으로 처리.
        이미지가 타일 크기보다 작을 경우 패딩 처리함.
        """
        model.eval()
        c, h, w = img_tensor.shape
        
        # 이미지가 타일 크기보다 작으면 패딩 추가
        pad_h = max(0, self.tile_size - h)
        pad_w = max(0, self.tile_size - w)
        
        if pad_h > 0 or pad_w > 0:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
        with torch.no_grad():
            tiles, coords = self.tile_image(img_tensor)
            processed_tiles = []
            
            for tile in tiles:
                tile = tile.unsqueeze(0).to(device)
                out = model(tile)
                processed_tiles.append(out.squeeze(0))
            
            processed_tiles = torch.stack(processed_tiles)
            
            # 패딩된 상태의 대상 형상
            new_h, new_w = img_tensor.shape[1], img_tensor.shape[2]
            target_shape_padded = (c, new_h * self.upscale, new_w * self.upscale)
            
            result = self.merge_tiles(processed_tiles, coords, target_shape_padded)
            
            # 패딩 제거 후 원래 크기의 배수로 크롭
            final_h, final_w = h * self.upscale, w * self.upscale
            return result[:, :final_h, :final_w]

if __name__ == "__main__":
    # 타일링 테스트
    tiler = PanoTiler(tile_size=64, overlap=16, upscale=2)
    dummy_pano = torch.randn(1, 200, 500)
    tiles, coords = tiler.tile_image(dummy_pano)
    print(f"생성된 타일 수: {len(tiles)}")
    print(f"좌표 예시: {coords[0]}")
    
    # 더미 복원 테스트
    dummy_processed = tiles * 1.5 # 가상의 모델 처리
    restored = tiler.merge_tiles(dummy_processed, coords, (1, 400, 1000))
    print(f"복원된 이미지 크기: {restored.shape}")
