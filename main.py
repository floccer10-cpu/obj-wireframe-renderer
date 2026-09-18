from model import Model
from PIL import Image
def main(w : int = 1024, h : int = 1024, s : float = 9.0, src : str = "model.obj", out : str = "img.png") -> int:
    model = Model(src)
    buf = model.rasterize(W = w, H = h, scale = s)
    img = Image.fromarray(buf)
    img.save(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())