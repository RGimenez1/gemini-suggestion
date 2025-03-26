import openai
import os
import json
import time
import re
from datetime import datetime
from dotenv import load_dotenv
from midiutil import MIDIFile
import pygame
import sounddevice as sd
import soundfile as sf
import numpy as np
import io
import threading
import traceback
import warnings

# Suppress deprecation warnings for stream_to_file
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Configuration ---
load_dotenv()
try:
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not openai_client.api_key:
        raise ValueError("OPENAI_API_KEY not found.")
except Exception as e:
    print(f"CRITICAL Error initializing OpenAI client: {e}"); exit()

SONGS_FOLDER = "songs"
TEMP_WAV_FOLDER = "temp"
RECORDING_FILENAME = "temp_user_recording.wav"
SAMPLE_RATE = 16000  # 16kHz for compatibility with most audio processing

# Updated TTS Configuration for latest model
TTS_MODEL = "gpt-4o-mini-tts"  # Use the latest TTS model
TTS_VOICE = "fable"  # Voice option - still using standard voice names
TTS_SPEED = 1.2  # Speed parameter (1.0 is normal)
TTS_RESPONSE_FORMAT = "mp3"  # Using mp3 for broad compatibility

STT_MODEL = "whisper-1"  # Use Whisper via the dedicated transcription endpoint
LLM_MODEL = "gpt-4o"  # Keep gpt-4o for main reasoning

# Voice detection parameters
ENERGY_THRESHOLD = 0.005  # Adjust based on microphone sensitivity
SILENCE_THRESHOLD = 1.8  # Seconds of silence before considering input complete
MAX_LISTEN_TIME = 15.0  # Maximum total listening time in seconds
SPEECH_PADDING = 0.5  # Seconds to keep recording after silence to catch trailing words

# --- Pygame Mixer Initialization ---
try:
    pygame.init()
    # Pre-initialize mixer with recommended buffer size for lower latency
    pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 1024)
    pygame.mixer.init()
    # Reserve Channel 0 for TTS, leave mixer.music for MIDI
    pygame.mixer.set_num_channels(8)  # Ensure enough channels exist
    pygame.mixer.Channel(0).set_volume(1.0)
    print("Pygame mixer initialized successfully.")
except pygame.error as e: print(f"CRITICAL Error initializing pygame mixer: {e}. Exiting."); exit()
except Exception as e: print(f"CRITICAL Unexpected error during Pygame init: {e}. Exiting."); exit()

# Create temp folder if needed
os.makedirs(TEMP_WAV_FOLDER, exist_ok=True)
os.makedirs(SONGS_FOLDER, exist_ok=True)

# --- Global State Tracking ---
music_is_playing = False
music_event = threading.Event()

# --- Helper Functions ---
def sanitize_filename(name):
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    sanitized = sanitized.replace(" ", "_")
    return sanitized[:100]

def get_chord_notes(chord_name, base_octave=4):
    note_map = {'C': 0, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
    sharp_map = {'C#':'Db', 'D#':'Eb', 'E#':'F', 'F#':'Gb', 'G#':'Ab', 'A#':'Bb', 'B#':'C'}
    intervals = { 'maj': [0, 4, 7], 'm': [0, 3, 7], 'dim': [0, 3, 6], 'aug': [0, 4, 8], 'sus4': [0, 5, 7], 'sus2': [0, 2, 7], 'maj7': [0, 4, 7, 11], 'm7': [0, 3, 7, 10], '7': [0, 4, 7, 10], 'dim7': [0, 3, 6, 9], 'm7b5': [0, 3, 6, 10], '6': [0, 4, 7, 9], 'm6': [0, 3, 7, 9], 'maj9': [0, 4, 7, 11, 14], 'm9': [0, 3, 7, 10, 14], '9': [0, 4, 7, 10, 14], '6/9': [0, 4, 7, 9, 14], 'm11': [0, 3, 7, 10, 14, 17], '11': [0, 7, 10, 14, 17], 'maj7#11':[0, 4, 7, 11, 18], 'maj13':[0, 4, 7, 11, 14, 21], 'm13': [0, 3, 7, 10, 14, 21], '13': [0, 4, 10, 14, 21], '7sus4':[0, 5, 7, 10], }
    original_chord_name = chord_name; root_str = chord_name; quality = None
    for q in sorted(intervals.keys(), key=len, reverse=True):
        if root_str.endswith(q): quality = q; root_str = root_str[:-len(q)]; break
    if not quality:
        if root_str.endswith('m'): quality = 'm'; root_str = root_str[:-1]
        elif root_str.endswith('dim'): quality = 'dim'; root_str = root_str[:-3]
        elif root_str.endswith('aug'): quality = 'aug'; root_str = root_str[:-3]
        else: quality = 'maj'
    root_note_name = None;
    if not root_str: print(f"Warning: Invalid chord format - missing root note in '{original_chord_name}'"); return []
    if len(root_str) > 1 and root_str[1] == '#': root_note_name = sharp_map.get(root_str[:2])
    elif len(root_str) > 1 and root_str[1] == 'b': root_note_name = root_str[:2]
    else: root_note_name = root_str[0]
    if root_note_name not in note_map: print(f"Warning: Could not parse valid root note from '{root_str}' in '{original_chord_name}'"); return []
    try:
        root_midi_base = note_map[root_note_name]; root_note = root_midi_base + (base_octave * 12)
        notes = [(root_note + i) for i in intervals[quality]]; return notes
    except Exception as e: print(f"Error calculating notes for '{original_chord_name}' (Quality: {quality}): {e}"); return []

# --- Music Thread for Monitoring Playback ---
def music_monitor_thread():
    global music_is_playing
    while True:
        if pygame.mixer.music.get_busy():
            music_is_playing = True
            time.sleep(0.1)  # Check every 100ms
        else:
            if music_is_playing:  # Just stopped playing
                music_is_playing = False
                music_event.set()  # Signal that music is done
            time.sleep(0.1)  # Still check when not playing

# Start the music monitor thread
music_monitor = threading.Thread(target=music_monitor_thread, daemon=True)
music_monitor.start()

# --- Tool Function: Music Generation/Playback ---
def play_music_with_pygame(music_params):
    """Generates, saves, and plays MIDI music. Sets music_is_playing flag and returns when done."""
    global music_is_playing
    music_event.clear()  # Reset event at the start
    
    print("\nReceived request to generate music...")
    skipped_chords = []; generated_notes_count = 0
    try:
        action_type = music_params.get("type", "").lower(); data = music_params.get("data", [])
        bpm = music_params.get("bpm"); beats_per_chord = music_params.get("beats_per_chord")
        instrument = music_params.get("instrument_program", 0); song_title = music_params.get("song_title", None)
        if not bpm: return "Error: BPM parameter missing."
        if action_type == "chords" and beats_per_chord is None: print("Warning: 'beats_per_chord' missing for chords. Using default 4."); beats_per_chord = 4
        if not data: return "Error: No musical data provided."
        midi_file = MIDIFile(1); track = 0; channel = 0;
        midi_file.addTrackName(track, 0, song_title or "AI Music Track"); midi_file.addTempo(track, 0, bpm)
        midi_file.addProgramChange(track, channel, 0, instrument)
        current_beat = 0; volume = 100
        if action_type == "chords":
            print(f"Generating MIDI for chords: {data} at {bpm} BPM, {beats_per_chord} beats/chord")
            chord_duration_beats = beats_per_chord
            for chord_name in data:
                notes_to_play = get_chord_notes(chord_name)
                if not notes_to_play: print(f"Skipping unparsable chord: '{chord_name}'"); skipped_chords.append(chord_name)
                else:
                    for note in notes_to_play: midi_file.addNote(track, channel, note, current_beat, chord_duration_beats, volume); generated_notes_count += 1
                    current_beat += chord_duration_beats
        elif action_type == "notes":
             print(f"Generating MIDI for notes sequence at {bpm} BPM")
             for note_event in data:
                 pitch = note_event.get('pitch'); start_beat = note_event.get('start', 0)
                 duration_beats = note_event.get('duration', 1); event_velocity = note_event.get('velocity', volume)
                 if pitch is not None and isinstance(pitch, int): midi_file.addNote(track, channel, pitch, start_beat, duration_beats, event_velocity); generated_notes_count += 1
                 else: print(f"Skipping invalid note event: {note_event}")
        else: return f"Error: Unknown music type '{action_type}'."
        if generated_notes_count == 0: 
            error_msg = "Error: Failed to generate any valid notes."
            if skipped_chords: error_msg += f" Skipped chords: {', '.join(skipped_chords)}"
            print(error_msg); return error_msg
            
        base_filename = sanitize_filename(song_title) if song_title else "ai_generated_music"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S"); filename = f"{base_filename}_{timestamp}.mid"
        filepath = os.path.join(SONGS_FOLDER, filename)
        try:
             with open(filepath, "wb") as output_file: midi_file.writeFile(output_file)
             print(f"MIDI file saved: {filepath}")
        except Exception as write_error: return f"Error: Could not save MIDI file: {write_error}"
        
        if not pygame.mixer.get_init(): return "Error: Audio mixer not ready for MIDI."
        try:
            # Stop any current playback
            if pygame.mixer.music.get_busy(): 
                pygame.mixer.music.stop(); pygame.mixer.music.unload(); time.sleep(0.1)
            
            pygame.mixer.music.load(filepath); 
            print(f"Playing MIDI file: {filepath}..."); 
            pygame.mixer.music.play()
            music_is_playing = True
            
            success_msg = f"Successfully initiated MIDI playback of '{filename}'."
            if skipped_chords: success_msg += f" Warning: Skipped unparsable chords: {', '.join(skipped_chords)}."
            print(success_msg)
            return success_msg
            
        except pygame.error as e: return f"Error: Failed to play MIDI file: {e}"
    except Exception as e: print(f"Unexpected error in play_music_with_pygame: {e}"); traceback.print_exc(); return f"Unexpected error during music processing: {e}"

# --- TTS using OpenAI API with latest model ---
def speak(text_to_speak):
    """Uses OpenAI TTS API with gpt-4o-mini-tts and Pygame (mixer channel 0) to speak the text."""
    global music_is_playing
    
    if not text_to_speak: print("TTS: No text."); return
    if not pygame.mixer.get_init(): print("TTS Error: Mixer not init."); return

    # If music is playing, wait for it to finish first
    if music_is_playing:
        print("\nWaiting for music to finish before speaking...")
        music_event.wait(timeout=30.0)  # Wait up to 30 seconds for music to finish
        music_event.clear()
    
    print(f"\nAssistant Speaking: {text_to_speak}")
    try:
        # Using the latest TTS API endpoint with gpt-4o-mini-tts
        response = openai_client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text_to_speak,
            speed=TTS_SPEED,
            response_format=TTS_RESPONSE_FORMAT,
            instructions="speak in a THICK french accent. Very thick."
        )
        audio_bytes = response.content
        audio_stream = io.BytesIO(audio_bytes)

        sound = pygame.mixer.Sound(audio_stream)
        channel = pygame.mixer.Channel(0)

        # Simple check if TTS channel is busy, wait briefly
        start_wait_time = time.time()
        while channel.get_busy():
             pygame.time.Clock().tick(10)
             if time.time() - start_wait_time > 5:  # Timeout waiting for channel
                  print("TTS Warning: Channel busy timeout, attempting to play anyway.")
                  channel.stop()  # Force stop previous sound
                  break

        channel.play(sound)

        # Wait ONLY for THIS TTS sound to finish
        start_play_time = time.time()
        while channel.get_busy():
            pygame.time.Clock().tick(30)  # Check less frequently to reduce CPU
            if time.time() - start_play_time > 30:  # Max 30 sec wait for TTS playback
                 print("TTS Warning: Playback took too long, stopping.")
                 channel.stop()
                 break

    except openai.APIError as api_err: print(f"OpenAI TTS API Error: {api_err}")
    except pygame.error as pg_err: print(f"Pygame Error playing TTS: {pg_err}")
    except Exception as e: print(f"Error during TTS: {e}"); traceback.print_exc()
    finally:
        # Clean up stream
        if 'audio_stream' in locals() and audio_stream:
             audio_stream.close()

# --- Simple audio energy detection function ---
def is_speech(audio_chunk, threshold=ENERGY_THRESHOLD):
    """Detect if audio chunk contains speech based on energy level."""
    # Convert to float and normalize
    if audio_chunk.dtype != np.float32:
        audio_float = audio_chunk.astype(np.float32) / np.iinfo(audio_chunk.dtype).max
    else:
        audio_float = audio_chunk
    
    # Calculate RMS energy
    energy = np.sqrt(np.mean(audio_float**2))
    return energy > threshold

# --- STT using Simple Energy Detection and OpenAI Transcription API ---
def listen_and_transcribe():
    """Records audio using energy detection to detect speech end, transcribes with OpenAI Whisper."""
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if not input_devices: 
            print("No input audio devices found.")
            return None
    except Exception as e: 
        print(f"Error querying audio devices: {e}")
        return None

    buffer = []  # To collect audio data
    is_speaking = False
    listen_start_time = time.time()
    last_voice_time = time.time()
    
    # Get device info to find blocksize
    device_info = sd.query_devices(None, 'input')
    blocksize = int(SAMPLE_RATE * 0.1)  # 100ms chunks
    
    try:
        print("\nListening... (Speak and I'll detect when you finish)")
        
        # Start streaming audio with callback
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', 
                           blocksize=blocksize,
                           callback=lambda indata, frames, time, status: buffer.append(indata.copy())):
            
            while True:
                # Check if we've been listening too long overall
                if time.time() - listen_start_time > MAX_LISTEN_TIME:
                    print(f"Exceeded maximum listening time of {MAX_LISTEN_TIME}s")
                    break
                
                # Check if we have at least one audio chunk
                if len(buffer) > 0:
                    # Take the most recent chunk for analysis
                    audio_chunk = buffer[-1]
                    
                    # Check if it contains speech
                    if is_speech(audio_chunk):
                        if not is_speaking:
                            is_speaking = True
                            print("Speech detected!")
                        last_voice_time = time.time()
                    
                    # If we've been speaking but now have silence exceeding our threshold,
                    # consider input complete
                    elif is_speaking and (time.time() - last_voice_time) > SILENCE_THRESHOLD:
                        print("Speech ended - processing...")
                        # Keep recording for a short padding period to catch trailing words
                        time.sleep(SPEECH_PADDING)
                        break
                
                # Short sleep to prevent CPU spinning
                time.sleep(0.05)
                
        # If we detected speech, process it
        if not is_speaking or len(buffer) < 3:  # If no significant speech was detected
            print("No speech detected.")
            return None
            
        # Combine all buffered audio into one array
        all_audio = np.vstack(buffer)
        
        # Save to in-memory wav file
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, all_audio, SAMPLE_RATE, format='WAV', subtype='PCM_16')
        wav_bytes.seek(0)  # Reset stream position
        
        print("Transcribing audio...")
        # Send to OpenAI for transcription
        response = openai_client.audio.transcriptions.create(
            model=STT_MODEL,
            file=(RECORDING_FILENAME, wav_bytes, "audio/wav")
        )
        
        transcription = response.text
        print(f"Transcription: '{transcription}'")
        return transcription.strip() if transcription else None
        
    except sd.PortAudioError as pa_err: 
        print(f"Sounddevice/PortAudio Error: {pa_err}. Check mic.")
        return None
    except openai.APIError as api_err: 
        print(f"OpenAI Transcription API Error: {api_err}")
        return None
    except Exception as e: 
        print(f"Error during listening/transcription: {e}")
        traceback.print_exc()
        return None
    finally:
        # Clean up BytesIO stream if it exists
        if 'wav_bytes' in locals() and wav_bytes:
            wav_bytes.close()

# --- OpenAI Interaction & Tool Handling ---
conversation_history = [ {"role": "system", "content": """You are an expert Artist Musician, Sir Martian, using OpenAI audio. Interact via voice with personality.
        1. **Input:** You receive user requests transcribed from audio (via Whisper).
        2. **Process:** Understand request. Decide IF music generation is needed via the `play_music` tool.
        3. **Response (Text):** Formulate your response *text* in a lively, engaging style with personality. Be conversational but concise.
        4. **Output (Speech):** Your text responses will be converted to speech using OpenAI TTS with a lively voice.
        5. **Tool Use:** Call `play_music` tool as needed with correct parameters.
        6. **Engaging Tone:** Speak with enthusiasm and warmth. Use varied intonation where appropriate.
        7. **Confirm & Play:** Announce clearly what you are about to play (e.g., "Magnifique! Playing the chords for 'Love Me Harder'..." or "Alors, let me create something jazzy for you...") *before* the tool is called (music starts).""" } 
        ]

tools = [ # Only the music generation tool
    {"type": "function", "function": {"name": "play_music", "description": "Generates, SAVES, and plays music (chords/notes) via MIDI.", "parameters": {"type": "object", "properties": {"type": {"type": "string", "enum": ["chords", "notes"]}, "data": {"type": "array", "items": {}}, "bpm": {"type": "integer"}, "beats_per_chord": {"type": "integer"}, "instrument_program": {"type": "integer"}, "song_title": {"type": "string"} }, "required": ["type", "data", "bpm"] }}}
]

def process_user_input_and_respond(user_input_text):
    """Sends user text to OpenAI LLM, handles tool calls, returns final text for TTS."""
    global conversation_history
    if not user_input_text: return None

    conversation_history.append({"role": "user", "content": user_input_text})

    try:
        print("\nThinking...")
        response = openai_client.chat.completions.create(
            model=LLM_MODEL, # gpt-4o for reasoning
            messages=conversation_history,
            tools=tools,
            tool_choice="auto",
            temperature=0.7  # Slightly higher temperature for more personality
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        ai_response_text_before_tool = response_message.content or ""

        if tool_calls:
            if ai_response_text_before_tool: try_speak_with_fallback(ai_response_text_before_tool)

            conversation_history.append(response_message)
            available_functions = {"play_music": play_music_with_pygame}
            tool_results = []

            for tool_call in tool_calls:
                function_name = tool_call.function.name; function_to_call = available_functions.get(function_name)
                function_response_content = f"Error: Tool '{function_name}' not found."
                if function_to_call:
                    try:
                        function_args = json.loads(tool_call.function.arguments); print(f"AI requests tool call: {function_name} with args: {function_args}")
                        function_response_content = function_to_call(function_args) # Music starts here
                        print(f"Tool execution result: {function_response_content}")
                    except json.JSONDecodeError as json_err: error_msg = f"Error decoding args: {json_err}. Args: {tool_call.function.arguments}"; print(error_msg); function_response_content = error_msg
                    except Exception as e: error_msg = f"Error executing tool {function_name}: {e}"; print(error_msg); traceback.print_exc(); function_response_content = error_msg
                else: print(f"Error: AI requested unknown tool '{function_name}'")
                tool_results.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response_content})

            conversation_history.extend(tool_results)
            print("Getting final AI confirmation...")
            second_response = openai_client.chat.completions.create(
                model=LLM_MODEL, 
                messages=conversation_history,
                temperature=0.7  # Keep consistent temperature
            )
            final_ai_text = second_response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": final_ai_text})
            return final_ai_text

        else: # No tool call
            conversation_history.append(response_message)
            return ai_response_text_before_tool

    except openai.APIError as api_err: print(f"OpenAI API Error: {api_err}"); return "Sorry, error connecting with OpenAI."
    except Exception as e: print(f"Error processing input/response: {e}"); traceback.print_exc(); return "Sorry, an unexpected error occurred."


# --- Main Loop ---
print(f"\nAI Music Assistant (OpenAI Voice v4 with gpt-4o-mini-tts) Initialized. MIDI files saved in '{SONGS_FOLDER}'.")

# Add fallback mechanism in case the gpt-4o-mini-tts model isn't available
def try_speak_with_fallback(text):
    """Try to speak with the primary model, fall back to tts-1 if needed."""
    global TTS_MODEL
    try:
        speak(text)
    except Exception as e:
        if "gpt-4o-mini-tts" in str(e) and TTS_MODEL == "gpt-4o-mini-tts":
            print(f"Error with gpt-4o-mini-tts model: {e}")
            print("Falling back to tts-1 model...")
            TTS_MODEL = "tts-1"
            try:
                speak(text)
            except Exception as fallback_e:
                print(f"Even fallback TTS failed: {fallback_e}")
        else:
            print(f"Error during speech: {e}")

try:
    # Attempt a short, silent audio playback to ensure mixer is really ready
    # Create a short silent sound
    silent_sound_data = np.zeros((int(SAMPLE_RATE * 0.1), 1), dtype=np.int16) # 0.1 sec silence
    silent_stream = io.BytesIO()
    sf.write(silent_stream, silent_sound_data, SAMPLE_RATE, format='WAV')
    silent_stream.seek(0)
    pygame.mixer.Sound(silent_stream).play()
    time.sleep(0.2) # Give it a moment
    silent_stream.close()
    print("Audio output check successful.")
    try_speak_with_fallback("Hey, this is Sir Martian. Lets play some earthly music today, shall we?")
except Exception as audio_init_err:
     print(f"CRITICAL ERROR during audio output check: {audio_init_err}")
     print("Please check your audio output device and pygame setup.")
     exit()


while True:
    try:
        # Wait for any music to finish before listening
        if music_is_playing:
            print("Music is playing. Waiting before listening...")
            music_event.wait(timeout=30.0)  # Wait up to 30 seconds for music to finish
            music_event.clear()

        user_text = listen_and_transcribe()  # Now uses simple energy detection to detect end of speech

        if user_text:
             if user_text.lower() in ['quit', 'exit', 'stop', 'goodbye']:
                 try_speak_with_fallback("Goodbye! It was a pleasure making music with you.")
                 break

             ai_final_text = process_user_input_and_respond(user_text)

             if ai_final_text:
                 try_speak_with_fallback(ai_final_text)

        else:
            print("(No input received or recognized)")

    except (EOFError, KeyboardInterrupt): print("\nExiting..."); try_speak_with_fallback("Exiting."); break
    except Exception as loop_err: print(f"ERROR in main loop: {loop_err}"); traceback.print_exc(); try_speak_with_fallback("Sorry, something went wrong."); time.sleep(1)

# --- Cleanup ---
if pygame.mixer.get_init():
    pygame.mixer.music.stop()
    pygame.mixer.stop()
    pygame.mixer.quit()
    print("Pygame mixer quit.")
if pygame.get_init():
     pygame.quit()
     print("Pygame quit.")

print("Exited.") 