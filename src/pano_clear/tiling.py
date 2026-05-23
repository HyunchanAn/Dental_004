import torch
import torch.nn.functional as F

class PanoTiler:
    """
    怨좏빐?곷룄 ?뚮끂?쇰쭏 ?대?吏瑜?????⑥쐞濡?遺꾪븷?섍퀬 ?⑹튂???대옒??
    寃쎄퀎硫댁쓽 遺?먯뿰?ㅻ윭???諛⑹??섍린 ?꾪빐 Overlapping 諛?Weighted Blending ?곸슜.
    """
    def __init__(self, tile_size=512, overlap=64, upscale=2):
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.upscale = upscale

    def tile_image(self, img):
        """
        ?대?吏瑜?以묒꺽?섎뒗 ??쇰줈 遺꾪븷.
        img: (C, H, W) Tensor
        """
        c, h, w = img.shape
        tiles = []
        coords = []

        for y in range(0, h - self.overlap, self.stride):
            for x in range(0, w - self.overlap, self.stride):
                # ?대?吏 寃쎄퀎瑜?踰쀬뼱?섏? ?딅룄濡?議곗젙
                y_start = min(y, h - self.tile_size)
                x_start = min(x, w - self.tile_size)
                
                tile = img[:, y_start:y_start+self.tile_size, x_start:x_start+self.tile_size]
                tiles.append(tile)
                coords.append((y_start, x_start))
                
                if x_start + self.tile_size >= w:
                    break
            if y_start + self.tile_size >= h:
                break

        return torch.stack(tiles), coords

    def merge_tiles(self, tiles, coords, target_shape):
        """
        ??쇰뱾???⑹퀜???꾩껜 ?대?吏 蹂듭썝.
        tiles: (N, C, T, T) Tensor (Processed tiles)
        coords: ??쇱쓽 ?쒖옉 醫뚰몴 由ъ뒪??(?먮낯 湲곗?)
        target_shape: (C, H_large, W_large)
        """
        c, h_large, w_large = target_shape
        output = torch.zeros(target_shape, device=tiles.device)
        weights = torch.zeros(target_shape, device=tiles.device)

        # ???釉붾젋?⑹쓣 ?꾪븳 ?덈룄??留덉뒪???앹꽦 (寃쎄퀎硫댁쓣 遺?쒕읇寃?
        mask = self._get_mask(self.tile_size * self.upscale).to(tiles.device)

        for i, (y, x) in enumerate(coords):
            y_up, x_up = y * self.upscale, x * self.upscale
            t_size_up = self.tile_size * self.upscale
            
            # ??쇱씠 ?대?吏 寃쎄퀎瑜?踰쀬뼱??寃쎌슦瑜??鍮꾪븳 ?щ씪?댁떛 怨꾩궛
            y_end = min(y_up + t_size_up, h_large)
            x_end = min(x_up + t_size_up, w_large)
            
            tile_h = y_end - y_up
            tile_w = x_end - x_up
            
            # ??쇨낵 留덉뒪?щ? 寃쎄퀎 ?ш린??留욊쾶 ?먮쫫 (Clips if necessary)
            curr_tile = tiles[i][:, :tile_h, :tile_w]
            curr_mask = mask[:tile_h, :tile_w]
            
            output[:, y_up:y_end, x_up:x_end] += curr_tile * curr_mask
            weights[:, y_up:y_end, x_up:x_end] += curr_mask

        return output / (weights + 1e-8)

    def _get_mask(self, size):
        """
        ????뚮몢由щ줈 媛덉닔濡??щ챸?댁????좏삎 肄붿궗??留덉뒪???앹꽦.
        """
        mask_1d = torch.linspace(0, 1, steps=self.overlap * self.upscale)
        mask_1d = 0.5 - 0.5 * torch.cos(mask_1d * 3.14159)
        
        full_mask = torch.ones((size, size))
        transition = self.overlap * self.upscale
        
        # ?곹븯醫뚯슦 寃쎄퀎 泥섎━
        full_mask[:transition, :] *= mask_1d.view(-1, 1)
        full_mask[-transition:, :] *= mask_1d.flip(0).view(-1, 1)
        full_mask[:, :transition] *= mask_1d.view(1, -1)
        full_mask[:, -transition:] *= mask_1d.flip(0).view(1, -1)
        
        return full_mask

    def process_large_image(self, model, img_tensor, device):
        """
        紐⑤뜽???ъ슜?섏뿬 ??⑸웾 ?대?吏 ?꾩껜瑜???쇰쭅 諛⑹떇?쇰줈 泥섎━.
        ?대?吏媛 ????ш린蹂대떎 ?묒쓣 寃쎌슦 ?⑤뵫 泥섎━??
        """
        model.eval()
        c, h, w = img_tensor.shape
        
        # ?대?吏媛 ????ш린蹂대떎 ?묒쑝硫??⑤뵫 異붽?
        pad_h = max(0, self.tile_size - h)
        pad_w = max(0, self.tile_size - w)
        
        if pad_h > 0 or pad_w > 0:
            # reflect ?⑤뵫? ?⑤뵫 ?ш린媛 ?낅젰 李⑥썝 ?ш린蹂대떎 ?묒븘???섎?濡??먮윭 諛⑹?瑜??꾪빐 replicate 紐⑤뱶瑜??곸슜?⑸땲??
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='replicate')
        
        with torch.no_grad():
            tiles, coords = self.tile_image(img_tensor)
            processed_tiles = []
            
            for tile in tiles:
                tile = tile.unsqueeze(0).to(device)
                out = model(tile)
                processed_tiles.append(out.squeeze(0))
            
            processed_tiles = torch.stack(processed_tiles)
            
            # ?⑤뵫???곹깭??????뺤긽
            new_h, new_w = img_tensor.shape[1], img_tensor.shape[2]
            target_shape_padded = (c, new_h * self.upscale, new_w * self.upscale)
            
            result = self.merge_tiles(processed_tiles, coords, target_shape_padded)
            
            # ?⑤뵫 ?쒓굅 ???먮옒 ?ш린??諛곗닔濡??щ∼
            final_h, final_w = h * self.upscale, w * self.upscale
            return result[:, :final_h, :final_w]

if __name__ == "__main__":
    # ??쇰쭅 ?뚯뒪??
    tiler = PanoTiler(tile_size=64, overlap=16, upscale=2)
    dummy_pano = torch.randn(1, 200, 500)
    tiles, coords = tiler.tile_image(dummy_pano)
    print(f"?앹꽦??????? {len(tiles)}")
    print(f"醫뚰몴 ?덉떆: {coords[0]}")
    
    # ?붾? 蹂듭썝 ?뚯뒪??
    dummy_processed = tiles * 1.5 # 媛?곸쓽 紐⑤뜽 泥섎━
    restored = tiler.merge_tiles(dummy_processed, coords, (1, 400, 1000))
    print(f"蹂듭썝???대?吏 ?ш린: {restored.shape}")
