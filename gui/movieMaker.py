"""
AMMPER Movie Maker
@Daniel Palacios

edited by Madeline Marous
"""

import os

from moviepy.editor import ImageClip, concatenate_videoclips


def movie_maker(
    image_folder, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15
):
    """Export the fifteen frames and return the absolute video filename.

    Each frame is displayed for one second. All clips are closed on success
    or failure so repeated exports do not retain rendering resources.
    """
    images = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15]
    output_path = os.path.abspath("visualization.mp4")
    clips = []
    combined = None
    try:
        for image in images:
            clips.append(ImageClip(os.path.join(image_folder, image)).set_duration(1))
        combined = concatenate_videoclips(clips, method="compose")
        combined.write_videofile(output_path, fps=24)
        return output_path
    finally:
        if combined is not None:
            combined.close()
        for clip in clips:
            clip.close()
