import cv2
import imageio
import numpy as np

class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def _overlay_text(self, frame, text):
        if text is None:
            return frame
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        overlay = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        pad = 6
        text = str(text)
        lines = text.splitlines() if text else [""]
        sizes = [cv2.getTextSize(line, font, scale, thickness) for line in lines]
        line_dims = [(s[0][0], s[0][1], s[1]) for s in sizes]                    
        text_w = max((d[0] for d in line_dims), default=0)
        text_h_total = sum((d[1] + d[2] for d in line_dims), 0)
        x0, y0 = 5, 5
        y = y0 + pad
        for (line, (w, h, baseline)) in zip(lines, line_dims):
            y = y + h
                    
            cv2.putText(
                overlay,
                line,
                (x0 + pad + 1, y + 1),
                font,
                scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                line,
                (x0 + pad, y),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            y = y + baseline
        return overlay

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env, overlay_text=None):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.render()
            frame = self._overlay_text(frame, overlay_text)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
