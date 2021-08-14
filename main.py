import tkinter as tk
from tkinter.constants import CENTER
import pyaudio
import wave
import speech_recognition as sr
import threading
import numpy as np
import librosa, librosa.display
import tensorflow.keras as keras

class Controller():
    def __init__(self, model):
        self.root = tk.Tk()
        self.root.title("Authenticacion")
        self.root.geometry('400x300')
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure([0,1,2,3], weight=1)
        self.duracion = 2 # seconds
        self.file_name = "audio.wav"
        self.audio = pyaudio.PyAudio()
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.CHUNK = 1024
        self.myModel = model
        self.topredict = None
        self.user = None

        self.label_record_status_text = tk.StringVar()
        self.label_result_aut_text = tk.StringVar()

        self.button_record = tk.Button(master=self.root, text="Grabando audio", command=self.record)
        self.button_record.grid(column=0, row=0, sticky=tk.EW)
        
        self.label_record_status = tk.Label(master=self.root, textvariable=self.label_record_status_text)
        self.label_record_status.grid(column=0, row=1, sticky=tk.EW)
        
        self.button_auth = tk.Button(master=self.root, text="Ingresar", command=self.recognizer)
        self.button_auth.grid(column=0, row=2, sticky=tk.EW)

        self.label_result_aut = tk.Label(master=self.root, textvariable=self.label_result_aut_text)
        self.label_result_aut.grid(column=0, row=3, sticky=tk.EW)

        self.root.mainloop()

    def record(self):
        self.label_record_status_text.set("Grabando")
        t = threading.Thread(target=self.startRecord)
        t.start()

    def startRecord(self):
        stream=self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate = self.RATE, input=True,frames_per_buffer=self.CHUNK)
        frames=[]

        for i in range(0,int(44100/1024*self.duracion)):
            data=stream.read(1024)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        self.audio.terminate()

        waveFile=wave.open(self.file_name,'wb')
        waveFile.setnchannels(2)
        waveFile.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(44100)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()

        self.label_record_status_text.set("Record successfull")

    def recognizer(self):
        self.label_result_aut_text.set("Verificando")
        self.formatFile()
        self.myModel.predict(self.topredict)
        predictions = self.myModel.predict(self.topredict)
        self.user = np.argmax(predictions[0])
        print(self.user)
        if self.user == 2:
            self.label_result_aut_text.set("Bienvenido Kleber")
        #t = threading.Thread(target=self.startRecognizer)
        #t.start()

    def startRecognizer(self):
        recog = sr.Recognizer()
        with sr.AudioFile("./" + self.file_name) as source:
            recorder_audio = recog.listen(source)

            try:
                text = recog.recognize_google(
                    recorder_audio,
                    language="es-ES"
                )
                if text == "atún" and self.user == 2:
                    self.label_result_aut_text.set("Autenticacion pasada")
                else:
                    self.label_result_aut_text.set("Contraseña incorrecta")
            except Exception as es:
                self.label_result_aut_text.set("Error in auth")

    def formatFile(self):
        signal, sample_rate = librosa.load(self.file_name, sr=44100)
        mfcc = librosa.feature.mfcc(signal, sample_rate, n_mfcc=13, n_fft=4096, hop_length=512)
        mfcc = mfcc.T
        self.topredict = np.array(mfcc.tolist())
        self.topredict = np.expand_dims(self.topredict, axis=0)

if __name__ == "__main__":
    app = Controller(None)
