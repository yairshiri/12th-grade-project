import dataSaver
import math
from shapely.geometry import polygon
from shapely.ops import nearest_points
import concurrent.futures


class Sensor:
    def __init__(self, pos, angle):
        self.angle = angle
        self.x = pos[0]
        self.y = pos[1]
        self.max_distance = dataSaver.DataSaver.max_distance
        self.line = polygon.LineString([pos, self.get_points()[2:]])
        self.instance = dataSaver.DataSaver.get_instance()

    def update_pos(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.line = polygon.LineString([pos, self.get_points()[2:]])

    def get_points(self):
        x1 = self.x
        y1 = self.y
        y2 = self.y + math.sin(math.radians(self.angle)) * self.max_distance
        x2 = self.x + math.cos(math.radians(self.angle)) * self.max_distance
        return x1, y1, x2, y2

    def closest_wall(self):
        min_dis = 1000
        intersects = self.instance.STRtree.query(self.line)
        p = polygon.Point(self.line.coords[0])
        ret = polygon.Point(self.get_points()[2:])
        for geom in intersects:
            inter = geom.intersection(self.line)
            if inter.is_empty:
                continue
            points = nearest_points(inter, p)
            # if there is no intersection, then the distance is always 0 so we check for it.
            dis = p.distance(points[0])
            if min_dis > dis > 0:
                ret = points[0]

            min_dis = min(min_dis, dis if dis > 0 else min_dis)
        return min_dis, ret


class SensorGroup:
    def __init__(self, **kwargs):
        if len(kwargs) == 2:
            x = kwargs.get("x")
            y = kwargs.get("y")
        else:
            x, y = kwargs.get("p")
        self.sensors = [Sensor((x, y), 0),
                        Sensor((x, y), 45),
                        Sensor((x, y), 90),
                        Sensor((x, y), 135),
                        Sensor((x, y), 180),
                        Sensor((x, y), 225),
                        Sensor((x, y), 270),
                        Sensor((x, y), 315)]
        self.x, self.y = x, y

    def get_info(self):
        ret = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for sensor in self.sensors:
                ret.append(executor.submit(sensor.closest_wall).result())
        return ret

    def reset(self):
        self.update(x=self.x, y=self.y)

    def update(self, **kwargs):
        if len(kwargs) == 2:
            x = kwargs.get("x")
            y = kwargs.get("y")
        else:
            x, y = kwargs.get("p")
        for sensor in self.sensors:
            sensor.update_pos((x, y))
