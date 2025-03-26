import openai
import os
import json
import time
import tempfile
import re
from datetime import datetime
from dotenv import load_dotenv
from midiutil import MIDIFile
import pygame

# --- Configuration ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("Error: OPENAI_API_KEY not found.")
    exit()

SONGS_FOLDER = "songs"

# --- Pygame Mixer Initialization ---
try:
    pygame.init()
    pygame.mixer.init()
    print("Pygame mixer initialized successfully.")
except pygame.error as e:
    print(f"Error initializing pygame or mixer: {e}. Playback may fail.")

# --- Helper Functions ---
def sanitize_filename(name):
    """Removes/replaces characters invalid for filenames."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    sanitized = sanitized.replace(" ", "_")
    return sanitized[:100]

# --- *** OVERHAULED Chord Parsing *** ---
def get_chord_notes(chord_name, base_octave=4):
    """
    Parses a wide variety of chord names and returns MIDI note numbers.
    Uses common voicings for extended chords.
    """
    note_map = {'C': 0, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5, 'Gb': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11}
    sharp_map = {'C#':'Db', 'D#':'Eb', 'E#':'F', 'F#':'Gb', 'G#':'Ab', 'A#':'Bb', 'B#':'C'}

    original_chord_name = chord_name # For logging

    # --- Comprehensive Chord Quality Definitions (Intervals from Root) ---
    # Using common voicings, often omitting 5ths or less critical notes in extensions
    # R=0, m3=3, M3=4, P4=5, A4=#4=6, P5=7, A5=#5=8, m6=8(enharmonic), M6=9,
    # d7=9(enharmonic), m7=10, M7=11, 8va=12, m9=13, M9=14, #9=15, P11=17, #11=18,
    # m13=20, M13=21
    intervals = {
        # Basic Triads
        'maj':  [0, 4, 7],        # Major Triad (also default if no other quality matches)
        'm':    [0, 3, 7],        # Minor Triad
        'dim':  [0, 3, 6],        # Diminished Triad
        'aug':  [0, 4, 8],        # Augmented Triad
        # Suspended
        'sus4': [0, 5, 7],
        'sus2': [0, 2, 7],
        # Sevenths
        'maj7': [0, 4, 7, 11],    # Major Seventh
        'm7':   [0, 3, 7, 10],    # Minor Seventh
        '7':    [0, 4, 7, 10],    # Dominant Seventh
        'dim7': [0, 3, 6, 9],     # Diminished Seventh (Fully diminished)
        'm7b5': [0, 3, 6, 10],    # Half-diminished (Minor Seventh Flat Five)
        # Sixths
        '6':    [0, 4, 7, 9],     # Major Sixth (Major Triad + Major 6th)
        'm6':   [0, 3, 7, 9],     # Minor Sixth (Minor Triad + Major 6th)
        # Ninths (Voiced often R-3-7-9 or R-7-9 + 3/5) - Using R-3(or b3)-7(or b7)-9
        'maj9': [0, 4, 7, 11, 14], # Major Ninth (Maj7 + 9)
        'm9':   [0, 3, 7, 10, 14], # Minor Ninth (m7 + 9)
        '9':    [0, 4, 7, 10, 14], # Dominant Ninth (Dom7 + 9)
        '6/9':  [0, 4, 7, 9, 14],  # 6/9 Chord (Maj Triad + 6 + 9)
        # Elevenths (Often omit 3rd/5th, esp. dominant 11ths) - Using R-7-9-11 or R-b7-9-11
        'm11':  [0, 3, 7, 10, 14, 17], # Minor Eleventh (m9 + 11) - Keeping full for now
        '11':   [0, 7, 10, 14, 17],    # Dominant Eleventh (Voicing: R-P5-m7-M9-P11, omitting 3rd)
        'maj7#11':[0, 4, 7, 11, 18],   # Major 7 Sharp 11 (Lydian sound, R-3-7-#11, omit 5)
        # Thirteenths (Often omit 5th, 9th, 11th) - Using R-3(or b3)-7(or b7)-13, maybe 9
        'maj13':[0, 4, 7, 11, 14, 21], # Major Thirteenth (Maj9 + 13) - Keeping 9
        'm13':  [0, 3, 7, 10, 14, 21], # Minor Thirteenth (m9 + 13) - Keeping 9
        '13':   [0, 4, 10, 14, 21],    # Dominant Thirteenth (Voicing: R-3-m7-M9-M13, omit 5, 11)
        # Other Common
        '7sus4':[0, 5, 7, 10],
        # Add more complex alterations/voicings here if needed (e.g., 7#9, 7b9, ...)
    }

    root_str = chord_name
    quality = None

    # --- Improved Parsing: Match longest quality suffix first ---
    # Sort known qualities by length, descending
    for q in sorted(intervals.keys(), key=len, reverse=True):
        if root_str.endswith(q):
            quality = q
            root_str = root_str[:-len(q)] # Remove quality suffix from root string
            break

    # If no specific quality matched, check for simple major/minor/dim/aug that aren't suffixes
    if not quality:
        if root_str.endswith('m'): quality = 'm'; root_str = root_str[:-1]
        elif root_str.endswith('dim'): quality = 'dim'; root_str = root_str[:-3]
        elif root_str.endswith('aug'): quality = 'aug'; root_str = root_str[:-3]
        else: quality = 'maj' # Default to major if nothing else fits

    # --- Parse Root Note (handling sharps/flats) ---
    root_note_name = None
    if not root_str: # Handle case where only quality was given (e.g., "maj7") - Invalid
         print(f"Warning: Invalid chord format - missing root note in '{original_chord_name}'")
         return []

    if len(root_str) > 1 and root_str[1] == '#':
        root_note_name = sharp_map.get(root_str[:2]) # Convert C# -> Db etc.
        if root_note_name is None:
             print(f"Warning: Unrecognized sharp root '{root_str[:2]}' in '{original_chord_name}'")
             return []
        # Check if anything follows the sharp (e.g. C#m) - should have been handled by quality parsing already
    elif len(root_str) > 1 and root_str[1] == 'b':
        root_note_name = root_str[:2] # e.g., Bb, Eb
    else:
        root_note_name = root_str[0] # Single letter root: C, D, E etc.

    # Final check if root exists in our map
    if root_note_name not in note_map:
         print(f"Warning: Could not parse valid root note from '{root_str}' in '{original_chord_name}'")
         return []

    # --- Calculate MIDI Notes ---
    try:
        root_midi_base = note_map[root_note_name]
        root_note = root_midi_base + (base_octave * 12)

        chord_intervals = intervals[quality]
        notes = [(root_note + i) for i in chord_intervals]
        # Optional: Add logging to see what was parsed
        # print(f"  Parsed '{original_chord_name}': Root='{root_note_name}', Quality='{quality}', Notes={notes}")
        return notes

    except KeyError: # Should not happen if root_note_name check passes, but safety first
        print(f"Internal Error: Parsed root '{root_note_name}' not in note_map for '{original_chord_name}'")
        return []
    except Exception as e:
        print(f"Error calculating notes for '{original_chord_name}' (Quality: {quality}): {e}")
        return []


# --- MIDI File Generation and Playback (Added Error Aggregation) ---
def play_music_with_pygame(music_params):
    """
    Generates MIDI, SAVES it, plays it, and returns DETAILED status/errors.
    """
    print("\nReceived request to play music...")
    skipped_chords = [] # Track chords that couldn't be parsed
    generated_notes_count = 0 # Track if any music was actually generated
    try:
        # --- Parameter Extraction and Validation ---
        action_type = music_params.get("type", "").lower()
        data = music_params.get("data", [])
        bpm = music_params.get("bpm")
        beats_per_chord = music_params.get("beats_per_chord")
        instrument = music_params.get("instrument_program", 0)
        song_title = music_params.get("song_title", None)

        if not bpm: return "Error: BPM parameter missing."
        if action_type == "chords" and beats_per_chord is None:
             print("Warning: 'beats_per_chord' missing for chords. Using default 4.")
             beats_per_chord = 4
        if not data: return "Error: No musical data provided."

        # --- MIDI File Setup ---
        midi_file = MIDIFile(1)
        track = 0
        channel = 0
        midi_file.addTrackName(track, 0, song_title or "AI Music Track")
        midi_file.addTempo(track, 0, bpm)
        midi_file.addProgramChange(track, channel, 0, instrument)

        # --- Note Generation ---
        current_beat = 0
        volume = 100
        if action_type == "chords":
            print(f"Generating MIDI for chords: {data} at {bpm} BPM, {beats_per_chord} beats/chord")
            chord_duration_beats = beats_per_chord
            for chord_name in data:
                notes_to_play = get_chord_notes(chord_name) # Use NEW parser
                if not notes_to_play:
                    print(f"Skipping unparsable chord: '{chord_name}'")
                    skipped_chords.append(chord_name) # Record skipped chord
                    # Decide whether to advance time or leave a gap. Let's leave a gap.
                else:
                    for note in notes_to_play:
                        midi_file.addNote(track, channel, note, current_beat, chord_duration_beats, volume)
                        generated_notes_count += 1
                    current_beat += chord_duration_beats # Advance time only if chord was played

        elif action_type == "notes":
             # (Note generation logic remains similar, ensure it increments generated_notes_count)
            print(f"Generating MIDI for notes sequence at {bpm} BPM")
            for note_event in data:
                pitch = note_event.get('pitch'); start_beat = note_event.get('start', 0)
                duration_beats = note_event.get('duration', 1); event_velocity = note_event.get('velocity', volume)
                if pitch is not None and isinstance(pitch, int):
                    midi_file.addNote(track, channel, pitch, start_beat, duration_beats, event_velocity)
                    generated_notes_count += 1
                else: print(f"Skipping invalid note event: {note_event}")
        else:
            return f"Error: Unknown music type '{action_type}'."

        if generated_notes_count == 0:
             error_msg = "Error: Failed to generate any valid notes."
             if skipped_chords: error_msg += f" Skipped chords: {', '.join(skipped_chords)}"
             print(error_msg)
             return error_msg

        # --- File Saving ---
        os.makedirs(SONGS_FOLDER, exist_ok=True)
        base_filename = sanitize_filename(song_title) if song_title else "ai_generated_music"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_filename}_{timestamp}.mid"
        filepath = os.path.join(SONGS_FOLDER, filename)

        try:
             with open(filepath, "wb") as output_file: midi_file.writeFile(output_file)
             print(f"MIDI file saved: {filepath}")
        except Exception as write_error:
             return f"Error: Could not save MIDI file: {write_error}"

        # --- Playback ---
        if not pygame.mixer.get_init(): return "Error: Audio mixer not ready."
        try:
            if pygame.mixer.music.get_busy(): pygame.mixer.music.stop(); pygame.mixer.music.unload(); time.sleep(0.1)
            pygame.mixer.music.load(filepath)
            print(f"Playing MIDI file: {filepath}...")
            pygame.mixer.music.play()

            # --- Construct Success/Warning Message ---
            success_msg = f"Successfully initiated playback of '{filename}'."
            if skipped_chords:
                 success_msg += f" Warning: Skipped unparsable chords: {', '.join(skipped_chords)}."
                 print(f"Warning during playback initiation: Skipped chords {skipped_chords}")
            return success_msg

        except pygame.error as e: return f"Error: Failed to play MIDI file: {e}"

    except Exception as e:
        print(f"Unexpected error in play_music_with_pygame: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return f"Unexpected error during music processing: {e}"

# --- OpenAI Interaction (Adjusted System Prompt) ---
conversation_history = [
    {
        "role": "system",
        "content": """You are an expert AI Music Assistant focused on accuracy and reliability. Your goal is to play recognizable versions of songs or musical ideas requested by the user via MIDI.
        1.  **Prioritize Accuracy:** When a specific song is requested, make a strong effort to find the standard chord progression(s) used in the actual song (verse, chorus, etc.). Your knowledge now includes extended chords (6, 7, 9, 11, 13, sus, dim, aug, m7b5, etc.). Use standard notation (e.g., 'Cmaj7', 'G7', 'Am7b5', 'F#m', 'Bb13').
        2.  **Use the Tool Correctly:** Call `play_music` when playback is needed. Provide:
            *   `type`: 'chords' or 'notes'.
            *   `data`: Accurate list of standard chord names or note objects.
            *   `bpm`: Appropriate tempo.
            *   `beats_per_chord`: Required for 'chords'.
            *   `instrument_program`: Suitable MIDI instrument number.
            *   `song_title`: The actual song title if applicable (for saving).
        3.  **Handle Feedback:** The tool will report success, or errors including specific chords it couldn't parse. If chords were skipped, acknowledge this and consider if you can use alternative, parsable chords next time if the user wants to try again.
        4.  **Be Clear:** Explain what you are playing (e.g., "Okay, playing the verse chords for 'Let It Be' on piano..."). Confirm playback initiation based on the tool's success message."""
    }
]

# Tool Definition remains the same as the previous corrected version
tools = [
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": "Generates, SAVES, and plays a piece of music via MIDI.",
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["chords", "notes"], "description": "Type of musical data ('chords' or 'notes')."},
                    "data": {"type": "array", "description": "Musical data: Array of chord strings or note objects.", "items": {}},
                    "bpm": {"type": "integer", "description": "Tempo in BPM."},
                    "beats_per_chord": {"type": "integer", "description": "Required if type='chords'. Duration per chord in beats."},
                    "instrument_program": {"type": "integer", "description": "Optional MIDI program number (default: 0 - Piano)."},
                    "song_title": {"type": "string", "description": "Optional song title for saving (e.g., 'Let It Be')."}
                },
                "required": ["type", "data", "bpm"]
            }
        }
    }
]

# get_ai_response_and_call_tool remains the same structure as the previous version
# (It correctly handles sending tool results, including error messages, back to the AI)
def get_ai_response_and_call_tool():
    global conversation_history
    try:
        print("Sending request to OpenAI...")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            tools=tools,
            tool_choice="auto",
            temperature=0.5, # Even lower temperature to encourage factual recall
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            conversation_history.append(response_message)
            available_functions = {"play_music": play_music_with_pygame}
            tool_results = []

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)
                function_response_content = f"Error: Tool '{function_name}' not found."

                if function_to_call:
                    try:
                        function_args = json.loads(tool_call.function.arguments)
                        print(f"AI requests tool call: {function_name} with args: {function_args}")
                        # Execute the tool function, getting back a status/error message
                        function_response_content = function_to_call(music_params=function_args)
                        print(f"Tool execution result: {function_response_content}")
                    except json.JSONDecodeError as json_err:
                        error_msg = f"Error decoding arguments for tool {tool_call.id}: {json_err}. Args: {tool_call.function.arguments}"
                        print(error_msg)
                        function_response_content = error_msg
                    except Exception as e:
                        error_msg = f"Error executing tool {function_name}: {e}"
                        print(error_msg)
                        import traceback
                        traceback.print_exc() # Print stack trace for unexpected errors in tool
                        function_response_content = error_msg
                else:
                    print(f"Error: AI requested unknown tool '{function_name}'")

                tool_results.append({
                    "tool_call_id": tool_call.id, "role": "tool",
                    "name": function_name, "content": function_response_content,
                })

            conversation_history.extend(tool_results)

            # Get follow-up response from AI
            print("Sending tool execution results back to OpenAI...")
            second_response = openai.chat.completions.create(model="gpt-4o", messages=conversation_history)
            final_ai_msg = second_response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": final_ai_msg})
            return final_ai_msg
        else: # No tool call
            ai_msg = response_message.content
            conversation_history.append({"role": "assistant", "content": ai_msg})
            return ai_msg

    except openai.APIError as api_err:
        print(f"OpenAI API Error: {api_err}")
        try: print(f"Error details: {api_err.response.json()}")
        except: pass
        return "Sorry, error connecting with OpenAI."
    except Exception as e:
        print(f"Error interacting with OpenAI or tools: {e}")
        import traceback
        traceback.print_exc()
        return "Sorry, an unexpected error occurred."


# --- Main Loop (remains the same) ---
print(f"\nAI Music Assistant Initialized. MIDI files saved in '{SONGS_FOLDER}'. Type 'quit' to exit.")
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == 'quit': break
        conversation_history.append({"role": "user", "content": user_input})
        ai_response_text = get_ai_response_and_call_tool()
        print(f"\nAssistant: {ai_response_text}")
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        break
# --- Cleanup ---
if pygame.mixer.get_init(): pygame.mixer.music.stop(); pygame.mixer.quit(); print("Pygame mixer quit.")
if pygame.get_init(): pygame.quit(); print("Pygame quit.")
print("Exited.")