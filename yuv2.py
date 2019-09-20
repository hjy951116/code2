FFMPEG_BIN = "ffmpeg" 
import subprocess as sp
command = [ FFMPEG_BIN,
         '-framerate', '25', '-s', '1280x720' ,'-pixel_format', 'yuv420p', '-i', 'Crew_1280x720_60Hz.yuv', 'frames%d.jpg']
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
# -video_size 1920x1080 -r 25 -pixel_format yuv420p -i akiyo.yuv output-%d.jpg
#  '-f', 'rawvideo', 
#  '-c' ,'copy' ,'-f','segment','-segment_time', '0.01',