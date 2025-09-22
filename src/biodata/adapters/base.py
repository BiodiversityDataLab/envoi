class BaseAdapter:
    def open(self, spec: dict):
        raise NotImplementedError

    def sample_point(self, lat: float, lon: float, when=None, window_m: int = 0) -> dict:
        raise NotImplementedError
