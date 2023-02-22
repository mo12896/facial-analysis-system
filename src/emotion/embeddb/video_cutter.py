import moviepy.editor as mp


def cut_video(input_path, output_path, start_time, end_time):
    video = mp.VideoFileClip(input_path)
    video_cut = video.subclip(start_time, end_time)
    video_cut.write_videofile(output_path)


if __name__ == "__main__":
    cut_video(
        "/home/moritz/Dropbox (MIT)/Cleaned_Team Data/team_13/2023-01-10/clip_0_10570_12012.mp4",
        "/home/moritz/Workspace/masterthesis/data/short_clip_debug.mp4",
        860,
        890,
    )
