import ctypes


class MouseController:
    def __init__(self):
        self.available = False
        self.error = None
        self.user32 = None
        self.screen_size = (0, 0)

        try:
            self.user32 = ctypes.windll.user32
            if hasattr(self.user32, "SetProcessDPIAware"):
                self.user32.SetProcessDPIAware()
            self.screen_size = (
                int(self.user32.GetSystemMetrics(0)),
                int(self.user32.GetSystemMetrics(1)),
            )
        except Exception as exc:
            self.error = f"Mouse control unavailable: {exc}"
            return

        if self.screen_size[0] <= 0 or self.screen_size[1] <= 0:
            self.error = "Mouse control unavailable: invalid screen size."
            return

        self.available = True

    def move_to(self, x, y):
        if not self.available or self.user32 is None:
            return False

        clamped_x = max(0, min(int(x), self.screen_size[0] - 1))
        clamped_y = max(0, min(int(y), self.screen_size[1] - 1))
        self.user32.SetCursorPos(clamped_x, clamped_y)
        return True
