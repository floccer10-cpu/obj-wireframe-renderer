from pathlib import Path
import numpy as np
from numba import njit
from PIL import Image

@njit(fastmath = True, cache = True)
def drawline(buf, x0, y0, x1, y1, rgb) -> None:
    dx = x1 - x0
    dy = y1 - y0
    steps = abs(dx) if abs(dx) > abs(dy) else abs(dy)
    if steps == 0:
        buf[int(y0), int(x0)] = rgb
        return
    xinc = dx/steps
    yinc = dy/steps

    for i in range(0, int(steps + 1)):
        buf[int(y0), int(x0)] = rgb
        x0 = x0 + xinc
        y0 = y0 + yinc

class Model:
    vertices : np.ndarray
    edges    : np.ndarray
    faces    : np.ndarray
    def __init__(self, path):
        # validating path
        p = self._validate_path(path)

        # parsing file
        self._parse_file(p)
            
        # faces to edges
        self._faces_to_edges()

    def rasterize(self, W : int, H : int, scale : float) -> np.ndarray:
        buf = np.zeros((H, W, 3), dtype=np.uint8)
        screen = self.vertices[:, :]*[W*scale, -H*scale] + [W/2, 0]
   
        for edge in self.edges:
            first, second = edge
            x0, y0 = screen[first]
            x1, y1 = screen[second]
            drawline(buf, x0, y0, x1, y1, [255, 255, 255])
        
        return buf

    def _faces_to_edges(self) -> None:
        self.edges = np.concatenate([
        self.faces[:, [0, 1]],
        self.faces[:, [1, 2]],
        self.faces[:, [0, 2]]
        ], axis = 0)
        self.edges = np.sort(self.edges, axis = 1)
        self.edges = np.unique(self.edges, axis = 0)

    @staticmethod
    def _validate_path(path : str | Path) -> Path:
        if not str(path).strip():
            raise ValueError("path can't be empty")
        p = Path(path)
        if not p.exists():
            raise ValueError("file doesn't exist")
        if not p.is_file():
            raise IsADirectoryError("not a file")
        suf = p.suffix
        if suf.lower() != ".obj":
            raise ValueError(f"expected .obj file, got {suf}")
        return p

    def _parse_file(self, p: Path) -> None:
        with p.open(mode='r', errors="replace", encoding="utf-8") as f:
            V: list[tuple[float, float]]  = []
            F: list[tuple[int, int, int]] = []
            for lineno, raw in enumerate(f, start = 1):
                parts = raw.split()
                if parts[0] == 'v':
                    if len(parts) < 4:
                        raise ValueError(f"{p}:{lineno}: vertex must have 3 coordinates")
                    try:
                        x, y = float(parts[1]),  float(parts[2])
                        V.append((x, y))
                    except ValueError as e:
                        raise ValueError(f"{p}:{lineno}: bad vertex: {e}") from e
           
                elif parts[0] == 'f':
                    idx: list[int] = []
                    for token in parts[1:]:
                        head = token.split("/", 1)[0] # первая координата - индекс вершины модели
                        if not head:
                            raise ValueError(f"{p}:{lineno}: empty face index")
                        try:
                            idx.append(int(head) - 1)
                        except ValueError as e:
                            raise ValueError(f"{p}:{lineno}: bad face index: {e}") from e
                    if len(idx) < 3:
                        raise ValueError(f"{p}:{lineno}: face must point to at least 3 indices")
                    if any(i < 0 for i in idx):
                        raise ValueError(f"{p}:{lineno}: face indices must be natural numbers")
                    for i in range(1, len(idx) - 1):
                        F.append((idx[0], idx[i], idx[i+1]))
            if not V:
                raise ValueError(f"{p}: no vertices found")
            if not F:
                raise ValueError(f"{p}: no faces found")

            self.vertices = np.asarray(V, dtype = np.float64)
            self.faces = np.asarray(F, dtype = np.uint32)
            if self.faces.max() >= self.vertices.shape[0]:
                raise ValueError(f"{p}: faces reference vertex {self.faces.max() + 1}, only {self.vertices.shape[0]} vertices defined")

