import numpy as np

class SignalExtractor:
    def __init__(self):
        # Multi-ROI signal storage (FYP standard)
        self.signals = {
            "forehead": {"r": [], "g": [], "b": []},
            "left_cheek": {"r": [], "g": [], "b": []},
            "right_cheek": {"r": [], "g": [], "b": []},
        }

    def extract_rgb(self, roi, name):
        if roi is None or roi.size == 0:
            return None

        roi = roi.astype(np.float32)
        mask = np.any(roi > 0, axis=2)

        if np.sum(mask) == 0:
            return None
        valid_pixels = roi[mask]
        mean_b = np.mean(valid_pixels[:, 0])
        mean_g = np.mean(valid_pixels[:, 1])
        mean_r = np.mean(valid_pixels[:, 2])

        #store ROI
        self.signals[name]["r"].append(mean_r)
        self.signals[name]["g"].append(mean_g)
        self.signals[name]["b"].append(mean_b)

        return mean_r, mean_g, mean_b

    def get_signals(self):
        return {
            "forehead": {
                "r": np.array(self.signals["forehead"]["r"]),
                "g": np.array(self.signals["forehead"]["g"]),
                "b": np.array(self.signals["forehead"]["b"]),
            },
            "left_cheek": {
                "r": np.array(self.signals["left_cheek"]["r"]),
                "g": np.array(self.signals["left_cheek"]["g"]),
                "b": np.array(self.signals["left_cheek"]["b"]),
            },
            "right_cheek": {
                "r": np.array(self.signals["right_cheek"]["r"]),
                "g": np.array(self.signals["right_cheek"]["g"]),
                "b": np.array(self.signals["right_cheek"]["b"]),
            }
        }