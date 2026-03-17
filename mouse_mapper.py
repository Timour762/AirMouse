class MouseMapper:
    def __init__(self, frame_width, frame_height, screen_width, screen_height):
        self.frame_width = max(1, int(frame_width))
        self.frame_height = max(1, int(frame_height))
        self.screen_width = max(1, int(screen_width))
        self.screen_height = max(1, int(screen_height))

    def map_point(self, point):
        if point is None:
            return None

        x, y = point
        normalized_x = self._normalize(x, self.frame_width)
        normalized_y = self._normalize(y, self.frame_height)

        screen_x = int(normalized_x * (self.screen_width - 1))
        screen_y = int(normalized_y * (self.screen_height - 1))
        return (screen_x, screen_y)

    def _normalize(self, value, source_size):
        if source_size <= 1:
            return 0.0

        clamped_value = max(0, min(int(value), source_size - 1))
        return clamped_value / (source_size - 1)
