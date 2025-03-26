import openai
import os
import json
import time
import tempfile
from dotenv import load_dotenv
from midiutil import MIDIFile
import pygame # For MIDI playback via file

# --- Configuration ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
    exit()

# --- Pygame Mixer Initialization ---
try:
    pygame.init()
    pygame.mixer.init()
    print("Pygame mixer initialized successfully.")
except pygame.error as e:
    print(f"Error initializing pygame or mixer: {e}")
    print("MIDI playback will likely fail.")

# --- Music Definitions (Helper) ---
def get_chord_notes(chord_name, base_octave=4):
    note_map = {'C': 0, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
    root_str = chord_name.replace('b', 'b').replace('#', '#')
    quality = ''

    # More robust parsing order
    if 'maj7' in root_str: quality = 'maj7'; root_str = root_str[:-4]
    elif 'm7' in root_str: quality = 'm7'; root_str = root_str[:-2]
    elif '7' in root_str: quality = '7'; root_str = root_str[:-1]
    elif 'm' in root_str: quality = 'm'; root_str = root_str[:-1]

    # Handle flats/sharps in root note name correctly
    root_note_base_str = root_str[0]
    accidental = ''
    if len(root_str) > 1:
        if root_str[1] == 'b': accidental = 'b'
        elif root_str[1] == '#': accidental = '#' # Add sharp handling if needed

    root_note_name = root_note_base_str + accidental

    try:
        # Calculate base MIDI note
        root_note = note_map[root_note_name] + (base_octave * 12)

        # Build chord based on quality
        if quality == 'maj7': return [root_note, root_note + 4, root_note + 7, root_note + 11]
        elif quality == 'm7': return [root_note, root_note + 3, root_note + 7, root_note + 10]
        elif quality == '7': return [root_note, root_note + 4, root_note + 7, root_note + 10]
        elif quality == 'm': return [root_note, root_note + 3, root_note + 7]
        else: return [root_note, root_note + 4, root_note + 7] # Major triad (default if no quality)

    except KeyError:
        print(f"Warning: Could not parse root note for '{chord_name}' (parsed as '{root_note_name}')")
        return []

# --- MIDI File Generation and Playback Function ("Tool Execution") ---
def play_music_with_pygame(music_params):
    """
    Generates a temporary MIDI file based on music_params and plays it using pygame.
    """
    print("\nReceived request to play music...")
    try:
        action_type = music_params.get("type", "").lower()
        data = music_params.get("data", [])
        bpm = music_params.get("bpm")
        beats_per_chord = music_params.get("beats_per_chord")
        instrument = music_params.get("instrument_program", 0)

        # Validate required parameters based on type
        if not bpm:
             print("Error: BPM missing in music parameters.")
             return
        if action_type == "chords" and beats_per_chord is None:
             print("Warning: 'beats_per_chord' missing for chords type. Using default of 4.")
             beats_per_chord = 4 # Assign a default

        if not data:
            print("No musical data provided by AI.")
            return

        # --- Create MIDI File using midiutil ---
        midi_file = MIDIFile(1)
        track = 0
        channel = 0
        time_offset = 0

        midi_file.addTrackName(track, time_offset, "AI Music Track")
        midi_file.addTempo(track, time_offset, bpm)
        midi_file.addProgramChange(track, channel, time_offset, instrument)

        current_beat = 0
        volume = 100

        if action_type == "chords":
            print(f"Generating MIDI for chords: {data} at {bpm} BPM, {beats_per_chord} beats/chord")
            chord_duration_beats = beats_per_chord
            for chord_name in data:
                notes_to_play = get_chord_notes(chord_name)
                if not notes_to_play:
                    print(f"Skipping unknown or unparsable chord: '{chord_name}'")
                    continue
                for note in notes_to_play:
                    midi_file.addNote(track, channel, note, current_beat, chord_duration_beats, volume)
                current_beat += chord_duration_beats

        elif action_type == "notes":
            print(f"Generating MIDI for notes sequence at {bpm} BPM")
            for note_event in data:
                pitch = note_event.get('pitch')
                start_beat = note_event.get('start', 0)
                duration_beats = note_event.get('duration', 1)
                event_velocity = note_event.get('velocity', volume)
                if pitch is not None and isinstance(pitch, int):
                    midi_file.addNote(track, channel, pitch, start_beat, duration_beats, event_velocity)
                else:
                     print(f"Skipping note event with invalid/missing pitch: {note_event}")
        else:
            print(f"Unknown music type specified by AI: {action_type}")
            return

        temp_midi_filepath = None # Initialize variable
        # --- Save MIDI to a temporary file ---
        try:
             with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as temp_midi_file:
                midi_file.writeFile(temp_midi_file)
                temp_midi_filepath = temp_midi_file.name
                print(f"Temporary MIDI file generated: {temp_midi_filepath}")
        except Exception as write_error:
             print(f"Error writing temporary MIDI file: {write_error}")
             return # Cannot proceed without the file

        # --- Play the MIDI file using pygame ---
        if not pygame.mixer.get_init():
             print("Error: Pygame mixer not initialized. Cannot play music.")
             if temp_midi_filepath and os.path.exists(temp_midi_filepath):
                 os.remove(temp_midi_filepath)
             return

        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                time.sleep(0.1)

            pygame.mixer.music.load(temp_midi_filepath)
            print(f"Playing MIDI file...")
            pygame.mixer.music.play()

            # Keep script alive while music plays? Optional.
            # while pygame.mixer.music.get_busy():
            #    time.sleep(0.1)
            # print("Playback appears finished.")

        except pygame.error as e:
            print(f"Error playing MIDI file with pygame: {e}")
        finally:
             # Attempt to clean up the temporary file
             if temp_midi_filepath:
                 try:
                     # Wait a bit longer before deleting
                     time.sleep(1.5) # Increased delay
                     if os.path.exists(temp_midi_filepath):
                          os.remove(temp_midi_filepath)
                          # print(f"Cleaned up temp file: {temp_midi_filepath}")
                 except OSError as e:
                     # This might happen if pygame still has a lock on it
                     print(f"Warning: Could not delete temporary MIDI file {temp_midi_filepath}: {e}")
                     print("It might be deleted automatically upon script exit if possible.")


    except Exception as e:
        print(f"Error in play_music_with_pygame: {e}")


# --- OpenAI Interaction ---
conversation_history = [
    {
        "role": "system",
        "content": """You are an AI Music Assistant running on Windows. Your goal is to chat with the user, understand their musical taste, and play music for them using a tool when appropriate.
        Keep track of the user's preferences. When you decide to play music, call the 'play_music' tool with the necessary parameters.
        Infer parameters like BPM, chords/notes, and style from the conversation. Use MIDI Program numbers for instruments (0: Piano, 24: Nylon Guitar, 32: Acoustic Bass, 40: Violin, etc.).
        If playing chords, provide a list of chord names (e.g., ["Fm7", "Ebmaj7", "Dbmaj7", "Cm7"]) AND the 'beats_per_chord' parameter.
        If playing notes, provide a list of note objects: [{"pitch": 60, "start": 0, "duration": 1, "velocity": 90}, ...], where 'start' and 'duration' are in beats.
        Be conversational, but call the tool when music playback is the primary action."""
    }
]

# --- FIX: Corrected Tool Definition ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": "Generates and plays a short piece of music via MIDI based on the provided parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["chords", "notes"],
                        "description": "The type of musical data to play ('chords' or 'notes')."
                    },
                    "data": {
                        "type": "array",
                        "description": "The musical data. If type is 'chords', an array of chord name strings (e.g., ['Cmaj7', 'Fm7']). If type is 'notes', an array of note objects (e.g., [{'pitch': 60, 'start': 0, 'duration': 1, 'velocity': 90}]).",
                        "items": {
                             # Specifies that the array contains items. Relying on description for exact format.
                        }
                    },
                    "bpm": {
                        "type": "integer",
                        "description": "The tempo in beats per minute (e.g., 120)."
                    },
                    "beats_per_chord": {
                        "type": "integer",
                        "description": "Required only if type is 'chords'. The duration of each chord in beats (e.g., 4)."
                    },
                     "instrument_program": {
                        "type": "integer",
                        "description": "Optional. MIDI program number for the instrument (default: 0 - Acoustic Grand Piano)."
                    }
                },
                "required": ["type", "data", "bpm"] # beats_per_chord checked in code
            }
        }
    }
]

def get_ai_response_and_call_tool():
    global conversation_history
    try:
        print("Sending request to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            tools=tools,
            tool_choice="auto",
            temperature=0.7,
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            conversation_history.append(response_message)
            available_functions = {"play_music": play_music_with_pygame}

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                if function_to_call:
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        print(f"AI requests tool call: {function_name} with args: {function_args}")
                        function_to_call(music_params=function_args)
                        function_response_content = f"Successfully initiated playback for {function_name}."
                        conversation_history.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response_content,
                            }
                        )
                    except json.JSONDecodeError as json_err:
                         print(f"Error: Could not decode arguments for tool call {tool_call.id}: {json_err}")
                         print(f"Received arguments: {tool_call.function.arguments}")
                         # Append error? Avoid loops. Just report and maybe don't send back to AI.
                    except Exception as e:
                        print(f"Error executing tool {function_name}: {e}")
                else:
                    print(f"Error: AI requested unknown tool '{function_name}'")

            # Get follow-up response from AI after tool execution
            print("Sending tool execution results back to OpenAI...")
            second_response = openai.chat.completions.create(
                model="gpt-4o",
                messages=conversation_history,
            )
            final_ai_msg = second_response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": final_ai_msg})
            return final_ai_msg

        else: # No tool call
            ai_msg = response_message.content
            conversation_history.append({"role": "assistant", "content": ai_msg})
            return ai_msg

    except openai.APIError as api_err: # Catch specific OpenAI errors
        print(f"OpenAI API Error: {api_err}")
        # Extract more details if possible
        try:
            error_details = api_err.response.json()
            print(f"Error details: {error_details}")
        except Exception:
            pass # Ignore if details aren't easily available
        return "Sorry, I had trouble connecting with the AI service."
    except Exception as e:
        print(f"Error interacting with OpenAI or tools: {e}")
        return "Sorry, I encountered an error processing that."


# --- Main Loop ---
print("\nAI Music Assistant Initialized (using Pygame for MIDI). Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    conversation_history.append({"role": "user", "content": user_input})

    ai_response_text = get_ai_response_and_call_tool()

    print(f"\nAssistant: {ai_response_text}")

# --- Cleanup ---
if pygame.mixer.get_init():
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    print("Pygame mixer quit.")
if pygame.get_init():
     pygame.quit()
     print("Pygame quit.")

print("Exiting.")