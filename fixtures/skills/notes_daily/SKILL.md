# Create a daily note in Apple Notes

Open Apple Notes, create a new note titled with today's date, and paste a fixed daily-planning template into the body.

## Parameters

- `note_title` (string, required, default: "2026-04-22") — title of the new note, typically today's date in ISO format.
- `note_template` (string, required) — template text pasted into the body of the note.
- `notes_folder` (string, optional, default: "Notes") — iCloud folder that receives the new note.

## Preconditions

- The Apple Notes app is installed and signed into the target iCloud account.
- The {notes_folder} folder exists in the Notes sidebar.

## Steps

1. Switch focus to the Apple Notes app and select the {notes_folder} folder in the sidebar.
2. Click the New Note button in the toolbar to create an empty note.
3. Type the note title on the first line: {note_title}
4. Press Return twice to start the body section, then paste the template: {note_template}

## Expected outcome

A new note with title {note_title} appears in {notes_folder}, and its body contains the {note_template} text, ready for the day's edits.

## Notes

Creating a new note is reversible (the note can be moved to Trash from within Notes), so no step is flagged destructive. The Runner should ensure the correct folder is selected in step 1.
