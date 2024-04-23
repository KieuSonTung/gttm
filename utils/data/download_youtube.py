from pytube import YouTube
import argparse
from pytube.cli import on_progress #this module contains the built in progress bar. 


def download(SAVE_PATH, link):
    try: 
        # object creation using YouTube 
        yt = YouTube(link, on_progress_callback=on_progress)
    except: 
        #to handle exception 
        print("Connection Error") 
    
    # # Get all streams and filter for mp4 files
    # mp4_streams = yt.streams.filter(file_extension='mp4').all()
    
    # # get the video with the highest resolution
    # d_video = mp4_streams[-1]
    
    try: 
        # downloading the video 
        yt.streams \
        .filter(progressive=True, file_extension='mp4') \
        .order_by('resolution') \
        .desc() \
        .first() \
        .download(output_path=SAVE_PATH)
        
        # d_video.download(output_path=SAVE_PATH)

        print('Video downloaded successfully!')
    except: 
        print("Some Error!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Youtube video')
    parser.add_argument('--dst', type=str,
                        help='Destination')
    parser.add_argument('--link', type=str,
                        help='link youtube')

    args = parser.parse_args()

    print("Start Download")
    download(args.dst, args.link)