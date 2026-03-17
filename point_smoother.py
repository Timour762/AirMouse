from collections import deque


class PointSmoother:
    def __init__(self, max_points):
        self.history = deque(maxlen=max(1, int(max_points)))

    def smooth(self, point):
        if point is None:
            return None

        self.history.append(point)
        x = int(sum(current_point[0] for current_point in self.history) / len(self.history))
        y = int(sum(current_point[1] for current_point in self.history) / len(self.history))
        return (x, y)

    def reset(self):
        self.history.clear()
