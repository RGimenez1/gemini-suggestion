import openai
import os
import json
import time
import tempfile
import re # Import regular expressions for sanitizing filenames
from datetime import datetime # For timestamp fallback
from dotenv import load_dotenv
from midiutil import MIDIFile
import pygame # For MIDI playback via file

# --- Configuration ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
    exit()

SONGS_FOLDER = "songs" # Define the folder name for saved MIDIs

# --- Pygame Mixer Initialization ---
try:
    pygame.init()
    pygame.mixer.init()
    print("Pygame mixer initialized successfully.")
except pygame.error as e:
    print(f"Error initializing pygame or mixer: {e}")
    print("MIDI playback will likely fail.")

# --- Helper Functions ---
def sanitize_filename(name):
    """Removes characters that are invalid for filenames."""
    # Remove invalid characters
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    # Replace spaces with underscores (optional, but common)
    sanitized = sanitized.replace(" ", "_")
    # Limit length (optional)
    return sanitized[:100] # Limit to 100 chars

# --- Music Definitions (Improved Helper) ---
def get_chord_notes(chord_name, base_octave=4):
    """
    Parses chord names (including sharps, m7b5, dominant 7ths)
    and returns a list of MIDI note numbers.
    """
    note_map = {'C': 0, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
    # Enharmonic equivalents for sharps
    sharp_map = {'C#':'Db', 'D#':'Eb', 'E#':'F', 'F#':'Gb', 'G#':'Ab', 'A#':'Bb', 'B#':'C'}

    original_chord_name = chord_name # Keep for logging
    root_str = chord_name
    quality = ''

    # Define intervals relative to root (0)
    intervals = {
        'maj7': [0, 4, 7, 11],
        'm7':   [0, 3, 7, 10],
        '7':    [0, 4, 7, 10],  # Dominant 7th (Major 3rd, Minor 7th) - CORRECTED
        'm7b5': [0, 3, 6, 10],  # Minor 7 flat 5 (Half-diminished) - ADDED
        'm':    [0, 3, 7],
        'maj':  [0, 4, 7],      # Explicit major triad
        'dim':  [0, 3, 6],      # Diminished triad
        # Add more qualities as needed (e.g., 'dim7': [0, 3, 6, 9])
    }

    # Parse quality first (longest match)
    # Sort interval keys by length descending to match e.g. 'maj7' before 'maj'
    for q in sorted(intervals.keys(), key=len, reverse=True):
        if root_str.endswith(q):
            quality = q
            root_str = root_str[:-len(q)]
            break
    if not quality: # Default to major triad if no quality found
        quality = 'maj'

    # Parse root note (handle sharps)
    root_note_name = root_str
    if len(root_str) > 1 and root_str[1] == '#':
        # Convert sharp to flat equivalent using sharp_map
        root_note_name = sharp_map.get(root_str[:2], None)
        if root_note_name is None:
             print(f"Warning: Could not find enharmonic equivalent for sharp '{root_str[:2]}' in chord '{original_chord_name}'")
             return []
        # Check if there are more characters after the sharp (e.g., C#maj7) - should have been handled by quality parsing
    elif len(root_str) > 1 and root_str[1] == 'b':
        root_note_name = root_str[:2] # Already a flat or natural like 'Bb'

    try:
        # Get the base MIDI note number for the root
        root_midi_base = note_map[root_note_name]
        root_note = root_midi_base + (base_octave * 12)

        # Calculate chord notes based on intervals
        chord_intervals = intervals[quality]
        notes = [(root_note + i) for i in chord_intervals]
        return notes

    except KeyError:
        print(f"Warning: Could not parse root note '{root_note_name}' derived from '{original_chord_name}'")
        return []
    except Exception as e:
        print(f"Error calculating notes for '{original_chord_name}': {e}")
        return []


# --- MIDI File Generation and Playback Function (Modified for Saving) ---
def play_music_with_pygame(music_params):
    """
    Generates a MIDI file based on music_params, SAVES it to the 'songs' folder,
    and plays it using pygame.
    """
    print("\nReceived request to play music...")
    try:
        action_type = music_params.get("type", "").lower()
        data = music_params.get("data", [])
        bpm = music_params.get("bpm")
        beats_per_chord = music_params.get("beats_per_chord")
        instrument = music_params.get("instrument_program", 0)
        song_title = music_params.get("song_title", None) # Get optional song title

        # Validate required parameters based on type
        if not bpm:
             print("Error: BPM missing in music parameters.")
             return "Error: BPM parameter missing." # Return error message for AI
        if action_type == "chords" and beats_per_chord is None:
             print("Warning: 'beats_per_chord' missing for chords type. Using default of 4.")
             beats_per_chord = 4

        if not data:
            print("No musical data provided by AI.")
            return "Error: No musical data provided."

        # --- Create MIDI File using midiutil ---
        midi_file = MIDIFile(1)
        track = 0
        channel = 0
        time_offset = 0

        midi_file.addTrackName(track, time_offset, song_title or "AI Music Track")
        midi_file.addTempo(track, time_offset, bpm)
        midi_file.addProgramChange(track, channel, time_offset, instrument)

        current_beat = 0
        volume = 100
        notes_added_count = 0 # Keep track if any notes were actually added

        if action_type == "chords":
            print(f"Generating MIDI for chords: {data} at {bpm} BPM, {beats_per_chord} beats/chord")
            chord_duration_beats = beats_per_chord
            for chord_name in data:
                notes_to_play = get_chord_notes(chord_name) # Use improved parser
                if not notes_to_play:
                    print(f"Skipping unknown or unparsable chord: '{chord_name}'")
                    continue
                for note in notes_to_play:
                    midi_file.addNote(track, channel, note, current_beat, chord_duration_beats, volume)
                    notes_added_count += 1
                current_beat += chord_duration_beats # Increment beat only if notes were found

        elif action_type == "notes":
            print(f"Generating MIDI for notes sequence at {bpm} BPM")
            for note_event in data:
                pitch = note_event.get('pitch')
                start_beat = note_event.get('start', 0)
                duration_beats = note_event.get('duration', 1)
                event_velocity = note_event.get('velocity', volume)
                if pitch is not None and isinstance(pitch, int):
                    midi_file.addNote(track, channel, pitch, start_beat, duration_beats, event_velocity)
                    notes_added_count += 1
                else:
                     print(f"Skipping note event with invalid/missing pitch: {note_event}")
        else:
            print(f"Unknown music type specified by AI: {action_type}")
            return f"Error: Unknown music type '{action_type}'."

        if notes_added_count == 0:
             print("No valid notes were added to the MIDI file. Aborting playback.")
             return "Error: Failed to generate any valid notes for playback."

        # --- Prepare Filename and Directory ---
        os.makedirs(SONGS_FOLDER, exist_ok=True) # Create 'songs' directory if needed

        base_filename = "ai_generated_music"
        if song_title:
            base_filename = sanitize_filename(song_title)

        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}_{timestamp}.mid"
        filepath = os.path.join(SONGS_FOLDER, filename)

        # --- Save MIDI to the 'songs' folder ---
        try:
             with open(filepath, "wb") as output_file:
                midi_file.writeFile(output_file)
                print(f"MIDI file saved: {filepath}")
        except Exception as write_error:
             print(f"Error writing MIDI file to {filepath}: {write_error}")
             return f"Error: Could not save MIDI file: {write_error}"

        # --- Play the MIDI file using pygame ---
        if not pygame.mixer.get_init():
             print("Error: Pygame mixer not initialized. Cannot play music.")
             # No need to delete the saved file here
             return "Error: Audio mixer not ready for playback."

        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                time.sleep(0.1)

            pygame.mixer.music.load(filepath) # Load the saved file
            print(f"Playing MIDI file: {filepath}...")
            pygame.mixer.music.play()

            # Return success message for AI tool response
            return f"Successfully initiated playback of '{filename}'."

        except pygame.error as e:
            print(f"Error playing MIDI file with pygame: {e}")
            return f"Error: Failed to play MIDI file: {e}"
        # No finally block needed to delete temp file anymore

    except Exception as e:
        print(f"Error in play_music_with_pygame: {e}")
        return f"Unexpected error during music playback setup: {e}"


# --- OpenAI Interaction ---
conversation_history = [
    {
        "role": "system",
        "content": """You are an AI Music Assistant running on Windows. Your primary goal is to accurately play requested songs or musical ideas for the user using MIDI.
        1.  **Understand Taste & Requests:** Chat with the user to understand their musical preferences and specific song requests.
        2.  **Prioritize Accuracy:** When asked to play a specific song, make your best effort to find and use the ACTUAL common chord progression(s) for that song (e.g., verse and chorus if possible). Avoid overly simplified or generic "inspired by" versions unless the user asks for that. Aim for recognizability.
        3.  **Tool Use:** When you decide music should be played, call the `play_music` tool.
        4.  **Tool Parameters:**
            *   Provide the correct `type` ('chords' or 'notes').
            *   Provide the accurate musical `data` (chord names list or note object list).
            *   Provide an appropriate `bpm`.
            *   If `type` is 'chords', ALWAYS provide `beats_per_chord`.
            *   Use appropriate MIDI `instrument_program` numbers (0: Piano, 24: Nylon Guitar, 32: Bass, 80/81: Synth Lead/Pad, etc.).
            *   **IMPORTANT:** If playing a specific song, include the `song_title` parameter in the tool call so the file can be saved correctly.
        5.  **Be Conversational:** Explain what you are about to play. Confirm after the tool call (which signifies playback has started)."""
    }
]

# --- Tool Definition (Added song_title) ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": "Generates, SAVES, and plays a piece of music via MIDI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["chords", "notes"],
                        "description": "The type of musical data ('chords' or 'notes')."
                    },
                    "data": {
                        "type": "array",
                        "description": "Musical data: Array of chord strings (['Cmaj7',...]) or note objects ([{'pitch': 60,...}]).",
                        "items": {} # Relies on description for item types
                    },
                    "bpm": {
                        "type": "integer",
                        "description": "Tempo in beats per minute."
                    },
                    "beats_per_chord": {
                        "type": "integer",
                        "description": "Required if type is 'chords'. Duration of each chord in beats."
                    },
                     "instrument_program": {
                        "type": "integer",
                        "description": "Optional. MIDI program number for instrument (default: 0 - Piano)."
                    },
                    "song_title": { # ADDED
                         "type": "string",
                         "description": "Optional. The title of the song being played, used for saving the file (e.g., 'Let It Be')."
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
            temperature=0.6, # Slightly lower temp might encourage more factual chord recall
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            # Append the AI's response message that *requests* the tool call
            conversation_history.append(response_message)
            available_functions = {"play_music": play_music_with_pygame}

            tool_results = [] # Store results for each tool call

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                function_response_content = f"Error: Unknown tool '{function_name}' requested." # Default error

                if function_to_call:
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        print(f"AI requests tool call: {function_name} with args: {function_args}")

                        # *** Execute the actual function (our MIDI player/saver) ***
                        # The function now returns a status message string
                        function_response_content = function_to_call(music_params=function_args)

                        print(f"Tool execution result: {function_response_content}")

                    except json.JSONDecodeError as json_err:
                         error_msg = f"Error decoding arguments for tool {tool_call.id}: {json_err}. Args: {tool_call.function.arguments}"
                         print(error_msg)
                         function_response_content = error_msg
                    except Exception as e:
                        error_msg = f"Error executing tool {function_name}: {e}"
                        print(error_msg)
                        function_response_content = error_msg
                else:
                    print(f"Error: AI requested unknown tool '{function_name}'")

                # Append the result of the tool execution to the history
                tool_results.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response_content, # Pass back the status/error message
                    }
                )

            # Append all tool results before getting the final AI response
            conversation_history.extend(tool_results)

            # === Get follow-up response from AI ===
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

    except openai.APIError as api_err:
        print(f"OpenAI API Error: {api_err}")
        try:
            error_details = api_err.response.json()
            print(f"Error details: {error_details}")
        except Exception: pass
        return "Sorry, I had trouble connecting with the AI service."
    except Exception as e:
        print(f"Error interacting with OpenAI or tools: {e}")
        return "Sorry, I encountered an error processing that."


# --- Main Loop ---
print(f"\nAI Music Assistant Initialized. MIDI files will be saved in '{SONGS_FOLDER}'. Type 'quit' to exit.")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        conversation_history.append({"role": "user", "content": user_input})

        ai_response_text = get_ai_response_and_call_tool()

        print(f"\nAssistant: {ai_response_text}")

    except EOFError: # Handle Ctrl+D or unexpected end of input
        print("\nExiting due to EOF.")
        break
    except KeyboardInterrupt: # Handle Ctrl+C
        print("\nExiting due to user interrupt.")
        break

# --- Cleanup ---
if pygame.mixer.get_init():
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    print("Pygame mixer quit.")
if pygame.get_init():
     pygame.quit()
     print("Pygame quit.")

print("Exiting.")