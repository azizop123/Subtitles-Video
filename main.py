import whisper
import os
import shutil
import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip
from tqdm import tqdm
from pydub import AudioSegment

# Set the path to your ffmpeg and ffprobe executables
AudioSegment.ffmpeg = "ffmpeg.exe"
AudioSegment.ffprobe = "ffprobe.exe"

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1  # Increased font scale for better visibility
FONT_THICKNESS = 4  # Increased thickness for better visibility

class VideoTranscriber:
    def __init__(self, model_path, video_path, audio_path):
        self.model = whisper.load_model(model_path)
        self.video_path = video_path
        self.audio_path = audio_path
        self.text_array = []
        self.fps = 0
        self.char_width = 0

    def convert_audio_to_wav(self):
        print('Converting audio to WAV format')
        audio = AudioSegment.from_file(self.audio_path)
        wav_audio_path = os.path.splitext(self.audio_path)[0] + '.wav'
        audio.export(wav_audio_path, format="wav")
        return wav_audio_path

    def transcribe_video(self):
        wav_audio_path = self.convert_audio_to_wav()
        print('Transcribing video')
        result = self.model.transcribe(wav_audio_path)
        text = result["segments"][0]["text"]
        textsize = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, frame = cap.read()
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.char_width = int(textsize[0] / len(text))

        for j in tqdm(result["segments"]):
            lines = []
            text = j["text"]
            end = j["end"]
            start = j["start"]
            total_frames = int((end - start) * self.fps)
            start_frame = int(start * self.fps)
            total_chars = len(text)
            words = text.split(" ")
            i = 0

            while i < len(words):
                words[i] = words[i].strip()
                if words[i] == "":
                    i += 1
                    continue
                length_in_pixels = (len(words[i]) + 1) * self.char_width
                remaining_pixels = width - length_in_pixels
                line = words[i]

                while remaining_pixels > 0:
                    i += 1
                    if i >= len(words):
                        break
                    length_in_pixels = (len(words[i]) + 1) * self.char_width
                    remaining_pixels -= length_in_pixels
                    if remaining_pixels < 0:
                        continue
                    else:
                        line += " " + words[i]

                line_array = [line, start_frame, start_frame + int(len(line) / total_chars * total_frames)]
                start_frame += int(len(line) / total_chars * total_frames)
                lines.append(line_array)
                self.text_array.append(line_array)

        cap.release()
        print('Transcription complete')

    def extract_frames(self, output_folder):
        print('Extracting frames')
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        N_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Properly maintain aspect ratio without incorrect cropping
            frame = cv2.resize(frame, (width, height))

            for i in self.text_array:
                if N_frames >= i[1] and N_frames <= i[2]:
                    text = i[0]
                    text_size, _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
                    text_x = int((frame.shape[1] - text_size[0]) / 2)
                    text_y = int(height - 150)  # Adjust the vertical position as needed
                    # Draw the text in white
                    cv2.putText(frame, text, (text_x, text_y), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
                    break

            cv2.imwrite(os.path.join(output_folder, f"{N_frames:06d}.jpg"), frame)
            N_frames += 1

        cap.release()
        print('Frames extracted')

    def create_video(self, output_video_path):
        print('Creating video')
        image_folder = os.path.join(os.path.dirname(self.video_path), "frames")
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        self.extract_frames(image_folder)

        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images.sort()

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        clip = ImageSequenceClip([os.path.join(image_folder, img) for img in images], fps=self.fps)
        audio = AudioFileClip(self.audio_path)
        clip = clip.set_audio(audio)
        clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
        shutil.rmtree(image_folder)


# Example usage
model_path = "base"  # Replace with your model path if necessary
video_path = "video.mp4"  # Provide the path to your video file
audio_path = "audio.mp3"  # Provide the path to your audio file
output_video_path = "output.mp4"  # Specify the output path for the final video

transcriber = VideoTranscriber(model_path, video_path, audio_path)
transcriber.transcribe_video()
transcriber.create_video(output_video_path)
