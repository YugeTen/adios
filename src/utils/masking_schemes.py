import torch, random, math

def segmask_to_box(m):
    box_mask = torch.zeros_like(m)
    if m.sum() != 0:
        idx_y = (m.sum(0) != 0).nonzero()
        idx_x = (m.sum(1) != 0).nonzero()
        x1, x2 = idx_x[0].item(), idx_x[-1].item()
        y1, y2 = idx_y[0].item(), idx_y[-1].item()
        box_mask[x1:x2, y1:y2] = 1
    else:
        box_mask = m
    return box_mask


def gen_mae_mask(m, mask_perc=0.75):
    mask = torch.ones_like(m)
    l = 16
    n = int(m.shape[-1]/l)
    n_observed = int((n*n)*(1-mask_perc))
    coord_observed = random.sample(range(n*n), n_observed)
    for coord in coord_observed:
        x = coord//n
        y = coord%n
        mask[x*l:x*l+l, y*l:y*l+l] = 0
    return mask

def gen_beit_mask(m, mask_perc=0.3):
    base_mask = torch.zeros_like(m)
    h, w = m.shape
    min_pixels = 64
    a = 0
    while base_mask.sum() < mask_perc*h*w:
        s = random.sample(range(min_pixels, max(min_pixels+1,int(h*w*mask_perc)-a)), 1)[0]
        r = random.random()*(1/0.3-0.3) + 0.3 # aspect ratio of the block
        a, b = int(math.sqrt(s*r)), int(math.sqrt(s/r))
        t, l = random.randrange(0, w-a), random.randrange(0, h-b)
        base_mask[t:t+a, l:l+b] = 1
    return base_mask