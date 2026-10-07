"""Video I/O and visualization, independent of models and prediction logic."""

import cv2


class VideoReader:
    """Context-managed reader yielding BGR frames and their source indices."""

    def __init__(self, path):
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open {path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self, start=0, step=1, limit=None):
        """Seek and yield every `step`th frame, up to `limit` output frames."""
        if start < 0 or step < 1 or (limit is not None and limit < 0):
            raise ValueError("start/limit must be nonnegative and step must be positive")
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        index = start
        yielded = 0
        while limit is None or yielded < limit:
            success, frame = self.capture.read()
            if not success:
                break
            if (index - start) % step == 0:
                yield index, frame
                yielded += 1
            index += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.capture.release()


class VideoWriter:
    """Context-managed writer; defaults to VP8 for notebook WebM playback."""

    def __init__(self, path, fps, size, codec="VP80"):
        self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise RuntimeError(f"Could not create the output video: {path}")

    def write(self, frame):
        self.writer.write(frame)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.writer.release()


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not open {path}")
    return image


def rgb_image(image):
    """Convert a BGR image for display in matplotlib."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_label(frame, text, origin, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.7, frame.shape[1] / 1400)
    thickness = max(2, int(scale * 2))
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = min(max(8, origin[0]), frame.shape[1] - width - 8)
    y = min(max(height + 12, origin[1]), frame.shape[0] - baseline - 8)
    cv2.rectangle(frame, (x - 6, y - height - 8), (x + width + 6, y + baseline + 5), (20, 20, 20), -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def annotate_frame(frame, boxes=(), labels=()):
    """Draw generic overlays on a copy, leaving the inference frame untouched.

    Boxes are (xyxy, BGR color, thickness); labels are (text, origin, BGR color).
    The caller decides which predictions to display and how to style them.
    """
    annotated = frame.copy()
    for box, color, thickness in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
    for text, origin, color in labels:
        draw_label(annotated, text, origin, color)
    return annotated
