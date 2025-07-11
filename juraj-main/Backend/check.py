import subprocess

# Check if ffmpeg command is available
try:
    result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        print("FFmpeg is working correctly!")
        print(result.stdout.decode())

    else:
        print("FFmpeg is not available.")
        print(result.stderr.decode())
except FileNotFoundError:
    print("FFmpeg is not installed or not found in the system path.")
