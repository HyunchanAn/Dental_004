import torch
import torch.nn.functional as F


class PanoTiler:
    """
    고해상도 파노라마 이미지를 타일 단위로 분할하고 합치는 클래스.
    경계면의 부자연스러움을 방지하기 위해 Overlapping 및 Cosine Weighted Blending 적용.
    """

    def __init__(self, tile_size=512, overlap=64, upscale=2):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.upscale = upscale

    def _get_mask(self, size):
        """
        가장자리로 갈수록 투명해지는 선형 코사인 마스크 생성.
        """
        mask_1d = torch.linspace(
            0, 1, steps=self.overlap * self.upscale, dtype=torch.float32
        )
        mask_1d = 0.5 - 0.5 * torch.cos(mask_1d * 3.14159)

        full_mask = torch.ones((size, size), dtype=torch.float32)
        transition = self.overlap * self.upscale

        # 상하좌우 경계 처리
        full_mask[:transition, :] *= mask_1d.view(-1, 1)
        full_mask[-transition:, :] *= mask_1d.flip(0).view(-1, 1)
        full_mask[:, :transition] *= mask_1d.view(1, -1)
        full_mask[:, -transition:] *= mask_1d.flip(0).view(1, -1)

        return full_mask

    def tile_image(self, img):
        """
        이미지를 중첩하는 타일로 분할. (가변 패딩이 선행 적용되었으므로 stride 단위로 정확히 떨어짐)
        img: (C, H, W) Tensor
        """
        c, h, w = img.shape
        tiles = []
        coords = []

        for y in range(0, h - self.overlap, self.stride):
            for x in range(0, w - self.overlap, self.stride):
                y_start = min(y, h - self.tile_size)
                x_start = min(x, w - self.tile_size)

                tile = img[
                    :,
                    y_start : y_start + self.tile_size,
                    x_start : x_start + self.tile_size,
                ]
                tiles.append(tile)
                coords.append((y_start, x_start))

                if x_start + self.tile_size >= w:
                    break
            if y_start + self.tile_size >= h:
                break

        return torch.stack(tiles), coords

    def merge_tiles(self, tiles, coords, target_shape):
        """
        타일들을 합쳐서 전체 이미지 복원 (코사인 블렌딩 방식 원복).
        tiles: (N, C, T, T) Tensor (Processed tiles)
        coords: 타일의 시작 좌표 리스트 (원본 기준)
        target_shape: (C, H_large, W_large)
        """
        c, h_large, w_large = target_shape
        output = torch.zeros(target_shape, dtype=torch.float32, device=tiles.device)
        weights = torch.zeros(target_shape, dtype=torch.float32, device=tiles.device)

        # 코사인 블렌딩을 위한 윈도우 마스크 생성
        mask = self._get_mask(self.tile_size * self.upscale).to(
            device=tiles.device, dtype=torch.float32
        )

        for i, (y, x) in enumerate(coords):
            y_up, x_up = y * self.upscale, x * self.upscale
            t_size_up = self.tile_size * self.upscale

            y_end = min(y_up + t_size_up, h_large)
            x_end = min(x_up + t_size_up, w_large)

            tile_h = y_end - y_up
            tile_w = x_end - x_up

            curr_tile = tiles[i][:, :tile_h, :tile_w]
            curr_mask = mask[:tile_h, :tile_w]

            output[:, y_up:y_end, x_up:x_end] += curr_tile * curr_mask
            weights[:, y_up:y_end, x_up:x_end] += curr_mask

        return output / (weights + 1e-8)

    def process_large_image(self, model, img_tensor, device):
        """
        모델을 사용하여 대용량 이미지 전체를 타일링 방식으로 처리.
        가변 해상도 패딩(Variable Resolution Padding) 로직 적용하여 코사인 매트릭스가 완벽히 정렬되도록 보장.
        """
        model.eval()
        c, h, w = img_tensor.shape

        # 코사인 마스크의 경계선 Fade-out 아티팩트를 방지하기 위해 Top, Left에도 최소 overlap만큼 패딩 적용
        pad_top = self.overlap
        pad_left = self.overlap

        h_temp = h + pad_top
        w_temp = w + pad_left

        # 가변 해상도 패딩 로직: 이미지가 stride 단위에 완벽히 정렬되도록 목표 해상도 계산
        # size = stride * K + overlap 형태가 되도록 하여 마지막 타일의 오차를 없앰
        target_h = (
            (h_temp - self.overlap + self.stride - 1) // self.stride
        ) * self.stride + self.overlap
        target_w = (
            (w_temp - self.overlap + self.stride - 1) // self.stride
        ) * self.stride + self.overlap

        # 이미지가 타일 사이즈보다 작은 경우 최소 tile_size 보장
        target_h = max(target_h, self.tile_size)
        target_w = max(target_w, self.tile_size)

        pad_bottom = target_h - h_temp
        pad_right = target_w - w_temp

        # reflect 모드를 통해 자연스러운 경계선 패딩 (Left, Right, Top, Bottom)
        img_padded = F.pad(
            img_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate"
        )

        with torch.no_grad():
            tiles, coords = self.tile_image(img_padded)

            # [Issue 2] 분할된 패치들을 단일 배치로 묶어 한 번의 포워드 패스로 처리 (Batch Tiling)
            tiles = tiles.to(device)
            processed_tiles = model(tiles)

            # 패딩이 적용된 전체 크기
            new_h, new_w = img_padded.shape[1], img_padded.shape[2]
            target_shape_padded = (c, new_h * self.upscale, new_w * self.upscale)

            # Cosine Blending을 이용한 병합
            result = self.merge_tiles(processed_tiles, coords, target_shape_padded)

            # 패딩 제거 및 원래 크기의 배수로 크롭 (슬라이싱)
            crop_top = pad_top * self.upscale
            crop_left = pad_left * self.upscale
            crop_bottom = crop_top + h * self.upscale
            crop_right = crop_left + w * self.upscale

            return result[:, crop_top:crop_bottom, crop_left:crop_right]


if __name__ == "__main__":
    # 타일링 테스트
    tiler = PanoTiler(tile_size=64, overlap=16, upscale=2)
    dummy_pano = torch.randn(1, 200, 500)
    tiles, coords = tiler.tile_image(dummy_pano)
    print(f"생성된 타일 수: {len(tiles)}")
    print(f"좌표 예시: {coords[0]}")

    # 더미 복원 테스트
    dummy_processed = tiles * 1.5
    restored = tiler.merge_tiles(
        dummy_processed, coords, (1, 400 + 16 * 2, 1000 + 16 * 2)
    )  # 대략적 크기
    print(f"복원된 이미지 크기: {restored.shape}")
