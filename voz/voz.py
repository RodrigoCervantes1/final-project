import tkinter as tk
from tkinter import messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
from PIL import Image, ImageTk
import time
import scipy.io.wavfile as wavfile
import speech_recognition as speech_recog
import io
import threading
from scipy.spatial.distance import cosine

class VoiceAuthApp:
    def __init__(self, master):
        self.master = master
        master.title("Autenticación por Voz")
        master.geometry("650x850")

        self.center_window()

        self.style = ttk.Style("litera")

        self.label = ttk.Label(master, text="Sistema de Autenticación por Voz", font=("TkDefaultFont", 16))
        self.label.pack(pady=20)

        # Opción 1: Grabar Frase
        self.option1_frame = ttk.LabelFrame(master, text="Opción 1: Grabar Frase")
        self.option1_frame.pack(pady=10, padx=10, fill="x")

        self.record_button = ttk.Button(self.option1_frame, text="Grabar Frase", command=self.record_phrase, style='primary.TButton')
        self.record_button.pack(pady=10)

        self.authenticate_button1 = ttk.Button(self.option1_frame, text="Autenticar con Frase", command=lambda: self.authenticate_with_live_text(1), style='success.TButton', state='disabled')
        self.authenticate_button1.pack(pady=10)

        # Opción 2: Grabar/Cargar Audio
        self.option2_frame = ttk.LabelFrame(master, text="Opción 2: Grabar/Cargar Audio")
        self.option2_frame.pack(pady=10, padx=10, fill="x")

        self.record_audio_button = ttk.Button(self.option2_frame, text="Grabar Audio", command=self.record_audio_file, style='warning.TButton')
        self.record_audio_button.pack(pady=10)

        self.load_audio_button = ttk.Button(self.option2_frame, text="Cargar Audio", command=self.load_audio_file, style='secondary.TButton')
        self.load_audio_button.pack(pady=10)

        self.authenticate_button2 = ttk.Button(self.option2_frame, text="Autenticar con Audio", command=lambda: self.authenticate_with_live_text(2), style='success.TButton', state='disabled')
        self.authenticate_button2.pack(pady=10)

        # Progress bar frame
        self.progress_frame = ttk.Frame(master)
        self.progress_frame.pack(pady=20, fill='x')

        self.progress = ttk.Progressbar(self.progress_frame, length=300, mode='determinate', style='info.Horizontal.TProgressbar')
        self.progress.pack()
        self.progress.pack_forget()  # Hide initially but space is reserved

        self.time_label = ttk.Label(self.progress_frame, text="")
        self.time_label.pack(pady=5)

        # Accuracy label frame
        self.accuracy_frame = ttk.Frame(master)
        self.accuracy_frame.pack(pady=5, fill='x')

        self.accuracy_label = ttk.Label(self.accuracy_frame, text="")
        self.accuracy_label.pack()
        self.accuracy_label.pack_forget()  # Hide initially but space is reserved

        # Lock frame (fixed position)
        self.lock_frame = ttk.Frame(master)
        self.lock_frame.pack(side='bottom', pady=20)

        self.lock_label = ttk.Label(self.lock_frame)
        self.lock_label.pack()

        self.toggle_lock_button = ttk.Button(self.lock_frame, text="Abrir Candado", command=self.toggle_lock, style='warning.TButton')
        self.toggle_lock_button.pack(pady=10)

        self.load_lock_images()

        self.saved_phrase = None
        self.saved_voice_features = None
        self.auth_voice_features = None
        self.loaded_voice_features = None
        self.model = self.create_model()

        self.recording_duration = 5  # duración de la grabación en segundos
        self.is_locked = True  # Estado inicial del candado
        self.recording_thread = None
        self.cancel_recording = False

        self.live_text_window = None
        self.live_text_label = None

        self.recognizer = speech_recog.Recognizer()

    def center_window(self):
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width - 650) // 2
        y = (screen_height - 850) // 2
        self.master.geometry(f"650x850+{x}+{y}")

    def load_lock_images(self):
        self.lock_closed = ImageTk.PhotoImage(Image.open("lock_closed.png").resize((100, 100)))
        self.lock_open = ImageTk.PhotoImage(Image.open("lock_open.png").resize((100, 100)))
        self.lock_label.config(image=self.lock_closed)

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(128,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def record_audio(self):
        fs = 44100
        self.master.after(0, self.show_progress_bar)
        self.cancel_recording = False
        recording = sd.rec(int(self.recording_duration * fs), samplerate=fs, channels=1, dtype='int16')

        for i in range(self.recording_duration * 10):
            if self.cancel_recording:
                sd.stop()
                self.master.after(0, lambda: self.time_label.config(text="Grabación cancelada"))
                self.master.after(0, self.close_live_text_window)
                return None, None
            self.master.after(0, lambda v=i: self.update_progress(v))
            time.sleep(0.1)

        sd.wait()
        self.master.after(0, lambda: self.time_label.config(text="Grabación finalizada"))

        try:
            audio_data = recording.flatten()

            byte_io = io.BytesIO()
            wavfile.write(byte_io, fs, audio_data)
            byte_io.seek(0)

            with speech_recog.AudioFile(byte_io) as source:
                audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio, language='es-ES')
            
            # Eliminar palabras duplicadas consecutivas sin completar palabras
            words = text.split()
            text = ' '.join(word for i, word in enumerate(words) if i == 0 or word.lower() != words[i-1].lower())
            
            self.master.after(0, lambda: self.update_live_text(text))
            time.sleep(2)  # Dar tiempo para que el usuario vea el texto final
            self.master.after(0, self.close_live_text_window)
            
            return audio_data, text
        except speech_recog.UnknownValueError:
            self.master.after(0, lambda: messagebox.showerror("Error", "No se pudo reconocer el audio"))
            return None, None
        except speech_recog.RequestError:
            self.master.after(0, lambda: messagebox.showerror("Error", "No se pudo conectar con el servicio de reconocimiento de voz"))
            return None, None
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", f"Ocurrió un error inesperado: {str(e)}"))
            return None, None
        finally:
            self.master.after(0, self.hide_progress_bar)

    def show_progress_bar(self):
        self.progress.pack()
        self.progress['value'] = 0

    def hide_progress_bar(self):
        self.progress.pack_forget()

    def update_progress(self, value):
        self.progress['value'] = (value + 1) * 100 / (self.recording_duration * 10)
        self.time_label.config(text=f"Tiempo restante: {self.recording_duration - value/10:.1f} segundos")

    def show_live_text_window(self):
        self.live_text_window = tk.Toplevel(self.master)
        self.live_text_window.title("Texto en tiempo real")
        self.live_text_window.geometry("400x100")
        self.live_text_label = ttk.Label(self.live_text_window, text="Escuchando...", wraplength=380)
        self.live_text_label.pack(expand=True, fill='both', padx=10, pady=10)
        # Iniciar un hilo para actualizar el texto en tiempo real
        threading.Thread(target=self.update_live_text_thread, daemon=True).start()

    def close_live_text_window(self):
        if self.live_text_window:
            self.live_text_window.destroy()
            self.live_text_window = None
            self.live_text_label = None

    def update_live_text_thread(self):
        r = speech_recog.Recognizer()
        error_printed = False
        last_words = []  # Lista para almacenar las últimas palabras reconocidas
        with speech_recog.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            while self.live_text_window:
                try:
                    audio = r.listen(source, timeout=0.5, phrase_time_limit=1)
                    text = r.recognize_google(audio, language='es-ES')
                    words = text.split()
                    # Filtrar palabras repetidas
                    new_words = [word for word in words if word not in last_words]
                    if new_words:
                        self.master.after(0, lambda w=new_words: self.update_live_text(" ".join(w)))
                        last_words = words[-3:]  # Mantener las últimas 3 palabras para comparar
                    error_printed = False
                except speech_recog.WaitTimeoutError:
                    pass
                except speech_recog.UnknownValueError:
                    self.master.after(0, lambda: self.update_live_text("..."))
                except Exception as e:
                    print(f"Error en reconocimiento de voz en tiempo real: {e}")
                    if not error_printed:
                        self.master.after(0, lambda: self.update_live_text("Error en reconocimiento de voz"))
                        error_printed = True

    def update_live_text(self, text):
        if self.live_text_label:
            current_text = self.live_text_label.cget("text")
            if current_text == "Escuchando...":
                new_text = text
            else:
                # Dividir el texto actual y el nuevo en palabras
                current_words = current_text.split()
                new_words = text.split()
                
                # Encontrar el punto de coincidencia
                match_point = 0
                for i in range(min(len(current_words), len(new_words))):
                    if current_words[-i-1].lower() != new_words[i].lower():
                        match_point = i
                        break
                
                # Combinar el texto actual con el nuevo, eliminando repeticiones
                new_text = " ".join(current_words[:-match_point] + new_words)
            
            self.live_text_label.config(text=new_text)
            self.live_text_window.update()

    def extract_features(self, audio):
        # Extract more comprehensive voice features
        mfccs = librosa.feature.mfcc(y=audio.astype(float), sr=44100, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio.astype(float), sr=44100)
        mel = librosa.feature.melspectrogram(y=audio.astype(float), sr=44100)
        contrast = librosa.feature.spectral_contrast(y=audio.astype(float), sr=44100)
        tonnetz = librosa.feature.tonnetz(y=audio.astype(float), sr=44100)
        
        features = np.hstack((
            np.mean(mfccs.T, axis=0),
            np.mean(chroma.T, axis=0),
            np.mean(mel.T, axis=0),
            np.mean(contrast.T, axis=0),
            np.mean(tonnetz.T, axis=0)
        ))
        return features

    def record_phrase(self):
        result = messagebox.askokcancel("Grabar Frase", f"Por favor, diga su frase de autenticación durante {self.recording_duration} segundos después de hacer clic en OK.")
        if not result:
            return
        self.show_live_text_window()
        self.recording_thread = threading.Thread(target=self.record_phrase_with_live_text)
        self.recording_thread.start()

    def record_phrase_with_live_text(self):
        audio, text = self.record_audio()
        if audio is None or text is None:
            return
        if self.show_confirmation_window(text):
            self.saved_phrase = text
            self.saved_voice_features = self.extract_features(audio)
            self.master.after(0, lambda: messagebox.showinfo("Grabación Exitosa", f"Se grabó exitosamente la frase: '{text}'"))
            self.master.after(0, lambda: self.authenticate_button1.config(state='normal'))
        else:
            self.master.after(0, lambda: messagebox.showinfo("Grabación Cancelada", "La grabación ha sido cancelada."))

    def authenticate_with_live_text(self, option):
        if self.saved_voice_features is None and self.loaded_voice_features is None:
            messagebox.showerror("Error", "Primero debe entrenar su voz o cargar un audio.")
            return

        if self.saved_phrase is None:
            messagebox.showerror("Error", "Primero debe grabar o cargar su frase de autenticación.")
            return

        if not self.is_locked:
            self.is_locked = True
            self.lock_label.config(image=self.lock_closed)
            self.toggle_lock_button.config(text="Abrir Candado")

        result = messagebox.askokcancel("Autenticar", f"Por favor, diga su frase de autenticación durante {self.recording_duration} segundos después de hacer clic en OK.")
        if not result:
            return

        self.show_live_text_window()
        self.recording_thread = threading.Thread(target=lambda: self.authenticate_thread(option))
        self.recording_thread.start()

    def authenticate_thread(self, option):
        audio, recognized_text = self.record_audio()
        if audio is None or recognized_text is None:
            return

        current_voice_features = self.extract_features(audio)

        # Comparar frases exactamente
        phrases_match = recognized_text.lower() == self.saved_phrase.lower()

        # Comparar características de voz
        if option == 1:  # Autenticar con Frase
            reference_features = self.saved_voice_features
        elif option == 2:  # Autenticar con Audio
            reference_features = self.loaded_voice_features if self.loaded_voice_features is not None else self.saved_voice_features

        voice_similarity = 1 - cosine(reference_features, current_voice_features)
        voice_accuracy = voice_similarity * 100

        if phrases_match and voice_similarity > 0.85:  # Ajusta este umbral según sea necesario
            self.is_locked = False
            self.master.after(0, lambda: self.lock_label.config(image=self.lock_open))
            self.master.after(0, lambda: self.toggle_lock_button.config(text="Cerrar Candado"))
            self.master.after(0, lambda: messagebox.showinfo("Éxito", f"Autenticación exitosa. Candado desbloqueado.\nFrase reconocida: '{recognized_text}'\nFrase guardada: '{self.saved_phrase}'\nSimilitud de voz: {voice_accuracy:.2f}%"))
        else:
            self.is_locked = True
            self.master.after(0, lambda: self.lock_label.config(image=self.lock_closed))
            self.master.after(0, lambda: self.toggle_lock_button.config(text="Abrir Candado"))
            self.master.after(0, lambda: messagebox.showerror("Error", f"Intento fallido. Las frases no coinciden o la voz no es similar.\nFrase reconocida: '{recognized_text}'\nFrase guardada: '{self.saved_phrase}'\nSimilitud de voz: {voice_accuracy:.2f}%"))

    def toggle_lock(self):
        self.is_locked = not self.is_locked
        if self.is_locked:
            self.lock_label.config(image=self.lock_closed)
            self.toggle_lock_button.config(text="Abrir Candado")
        else:
            self.lock_label.config(image=self.lock_open)
            self.toggle_lock_button.config(text="Cerrar Candado")

    def record_audio_file(self):
        result = messagebox.askokcancel("Grabar Audio", f"Se grabará un audio de {self.recording_duration} segundos. Haga clic en OK para comenzar.")
        if not result:
            return
        self.show_live_text_window()
        self.recording_thread = threading.Thread(target=self.record_audio_file_with_live_text)
        self.recording_thread.start()

    def record_audio_file_with_live_text(self):
        audio, text = self.record_audio()
        if audio is None or text is None:
            return
        if self.show_confirmation_window(text):
            file_path = filedialog.asksaveasfilename(defaultextension=".wav")
            if file_path:
                wavfile.write(file_path, 44100, audio.astype(np.int16))  # Guardar como PCM/LPCM
                self.master.after(0, lambda: messagebox.showinfo("Grabación Exitosa", f"Audio guardado en {file_path}"))
                self.load_audio_file(file_path)

    def load_audio_file(self, file_path=None):
        if file_path is None:
            file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.flac;*.ogg")])
        if file_path:
            audio, sample_rate = librosa.load(file_path, sr=None)
            features = self.extract_features(audio)
            self.loaded_voice_features = features

            try:
                with speech_recog.AudioFile(file_path) as source:
                    audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data, language='es-ES')
                    # Eliminar palabras duplicadas consecutivas
                    words = text.split()
                    text = ' '.join(word for i, word in enumerate(words) if i == 0 or word.lower() != words[i-1].lower())
                    self.saved_phrase = text
            except (speech_recog.UnknownValueError, speech_recog.RequestError, ValueError):
                self.master.after(0, lambda: messagebox.showerror("Error", "No se pudo reconocer el texto en el archivo de audio"))
                return

            self.master.after(0, lambda: messagebox.showinfo("Carga Exitosa", f"Audio cargado desde {file_path}\nTexto reconocido: {self.saved_phrase}"))
            self.master.after(0, lambda: self.authenticate_button2.config(state='normal'))

    def show_confirmation_window(self, text):
        confirmation = tk.Toplevel(self.master)
        confirmation.title("Confirmar Grabación")
        confirmation.geometry("400x150")
        
        label = ttk.Label(confirmation, text=f"Frase reconocida:\n{text}", wraplength=380)
        label.pack(pady=10)
        
        button_frame = ttk.Frame(confirmation)
        button_frame.pack(pady=10)
        
        result = tk.BooleanVar()
        result.set(False)
        
        save_button = ttk.Button(button_frame, text="Guardar", command=lambda: [result.set(True), confirmation.destroy()])
        save_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = ttk.Button(button_frame, text="Cancelar", command=lambda: confirmation.destroy())
        cancel_button.pack(side=tk.LEFT, padx=10)
        
        confirmation.protocol("WM_DELETE_WINDOW", lambda: confirmation.destroy())
        confirmation.transient(self.master)
        confirmation.grab_set()
        confirmation.wait_window()
        
        return result.get()

def main():
    root = ttk.Window()
    app = VoiceAuthApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()