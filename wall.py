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

    def normal_to_wall(self, rect):
        img_size = dataSaver.DataSaver.get_instance().wall_img.get_size()
        new_rect = (
            int(rect[0] * img_size[0]), int(rect[1] * img_size[1]), int(rect[2] * img_size[0]),
            int(rect[3] * img_size[1]))

        ret = []
        for i in range(new_rect[0], new_rect[2] + new_rect[0], img_size[0]):
            for j in range(new_rect[1], new_rect[3] + new_rect[1], img_size[1]):
                ret.append((i, j))
        return ret, new_rect

    def intersects(self, geom):
        return self.rect.intersects(geom)

    def locs(self):
        size = dataSaver.DataSaver.get_instance().draw_scaler
        for x in range(self.width):
            for y in range(self.height):
                yield (self.x1 + x) * size[0], (self.y1 + y) * size[1]
