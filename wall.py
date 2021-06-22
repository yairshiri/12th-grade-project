import dataSaver
from shapely.geometry import polygon


class Wall:
    def __init__(self, **kwargs):
        self.x1, self.y1 = kwargs.get("p")
        self.height = kwargs.get("height")
        self.width = kwargs.get("width")
        self.rect = polygon.Polygon([
            (self.x1, self.y1),
            (self.x1 + self.width, self.y1),
            (self.x1 + self.width, self.y1 + self.height),
            (self.x1, self.y1 + self.height)])

    def intersects(self, geom):
        return self.rect.intersects(geom)

    def locs(self):
        size = dataSaver.DataSaver.get_instance().draw_scaler
        for x in range(self.width):
            for y in range(self.height):
                yield (self.x1 + x) * size[0], (self.y1 + y) * size[1]
