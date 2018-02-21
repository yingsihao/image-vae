from ..image import combine_images


def combine_videos(videos, num_columns):
    num_frames = len(videos[0])
    return [combine_images([video[k] for video in videos], num_columns) for k in range(num_frames)]
