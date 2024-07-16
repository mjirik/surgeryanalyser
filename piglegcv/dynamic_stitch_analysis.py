import numpy as np
from scipy.interpolate import interp1d
from typing import Optional




def get_subsegment_of_tracks_points(tracks_points: dict, start_frame:Optional[int], stop_frame: Optional[int]) -> dict:
    """Take smaller part of tracks_points related to the segment betwen start_frame and stop_frame."""
    frame_ids = tracks_points["frame_ids"]
    data_pixels = tracks_points["data_pixels"]

    new_tracks_points = {
        "frame_ids": [None] * len(frame_ids),
        "data_pixels": [None] * len(data_pixels)
    }


    for tool_id in range(len(frame_ids)):
        new_tracks_points["frame_ids"][tool_id] = []
        new_tracks_points["data_pixels"][tool_id] = []

        for frame_id, data_px in zip(frame_ids[tool_id], data_pixels[tool_id]):
            if frame_id >= start_frame and frame_id <= stop_frame:
                new_tracks_points["frame_ids"][tool_id].append(frame_id)
                new_tracks_points["data_pixels"][tool_id].append(data_px)

            if frame_id > stop_frame:
                break

    return new_tracks_points


class Interpolation:
    def __init__(self, frame_ids, coordinates, fps):
        self.frame_ids = np.array(frame_ids)
        self.coordinates = np.array(coordinates)
        self.fps = fps

        # Interpolators for x and y coordinates
        self.interpolator_x = interp1d(self.frame_ids, self.coordinates[:, 0], kind='linear', fill_value='extrapolate')
        self.interpolator_y = interp1d(self.frame_ids, self.coordinates[:, 1], kind='linear', fill_value='extrapolate')

    def value_in_frame(self, frame_id, ignore_no_data_for_s=1.):
        """
        Returns the interpolated value of the coordinates in the specified frame_id.
        If there are no data for the specified frame_id, it will return (np.nan, np.nan)

        ignore_no_data_for_s: float - number of seconds to ignore if there are no data for the specified frame_id
        """
        frame_sensitivity = ignore_no_data_for_s * self.fps

        # Check if there are any frame IDs within the specified frame_sensitivity of the requested frame_id
        if not np.any(
                (self.frame_ids >= frame_id - frame_sensitivity) & (self.frame_ids <= frame_id + frame_sensitivity)):
            return (np.nan, np.nan)

        x = self.interpolator_x(frame_id)
        y = self.interpolator_y(frame_id)
        return (x, y)


class InstrumentDistance:
    def __init__(self, frame_ids1, coordinates1, frame_ids2, coordinates2, fps):
        self.instrument1 = Interpolation(frame_ids1, coordinates1, fps)
        self.instrument2 = Interpolation(frame_ids2, coordinates2, fps)
        self.fps = fps

        # Determine the common range of frame_ids
        self.min_frame = max(min(frame_ids1), min(frame_ids2))
        self.max_frame = min(max(frame_ids1), max(frame_ids2))

    def average_distance(self, ignore_no_data_for_s=1):

        total_distance = 0
        valid_frame_count = 0

        for frame_id in range(self.min_frame, self.max_frame + 1):
            coord1 = self.instrument1.value_in_frame(frame_id, ignore_no_data_for_s)
            coord2 = self.instrument2.value_in_frame(frame_id, ignore_no_data_for_s)

            if not np.isnan(coord1).any() and not np.isnan(coord2).any():
                distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
                total_distance += distance
                valid_frame_count += 1

        if valid_frame_count == 0:
            return np.nan  # No valid frames to calculate average distance

        return total_distance / valid_frame_count

    def frames_below_threshold(self, threshold, seconds_sensitivity=1):
        count = 0

        for frame_id in range(self.min_frame, self.max_frame + 1):
            coord1 = self.instrument1.value_in_frame(frame_id, seconds_sensitivity)
            coord2 = self.instrument2.value_in_frame(frame_id, seconds_sensitivity)

            if not np.isnan(coord1).any() and not np.isnan(coord2).any():
                distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
                if distance < threshold:
                    count += 1

        return count

    def seconds_below_threshold(self, threshold, seconds_sensitivity=1):
        frame_count = self.frames_below_threshold(threshold, seconds_sensitivity)
        return frame_count / self.fps
