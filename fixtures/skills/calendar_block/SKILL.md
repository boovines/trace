# Create a focus block in Google Calendar

Open Google Calendar in Chrome, pick a time slot on tomorrow's calendar, and create a titled focus block of the desired duration.

## Parameters

- `block_title` (string, required, default: "Focus block") — title text used for the calendar event.
- `block_start_time` (string, required, default: "2:00 PM") — display-format start time of the block (for example, "2:00 PM" or "14:00").
- `block_duration_minutes` (integer, optional, default: 30) — length of the focus block, in minutes.

## Preconditions

- Google Chrome is installed and signed into the target Google account.
- Tomorrow's date is reachable from the current Google Calendar view.

## Steps

1. Switch focus to Google Chrome and open Google Calendar in the active tab.
2. Navigate to tomorrow's day view so the target start time {block_start_time} is visible.
3. Click the {block_start_time} time cell in the day grid to open the new-event popover.
4. Type the event title into the popover: {block_title}
5. Adjust the event duration to {block_duration_minutes} minutes using the end-time picker.
6. Click the Save button to commit the event to your calendar.

## Expected outcome

A new event titled {block_title} appears on tomorrow's calendar starting at {block_start_time} and spanning {block_duration_minutes} minutes. The event is visible in both day and week views.

## Notes

Saving a calendar event is reversible (the event can be deleted afterwards), so no step is flagged destructive. The Runner should still verify the popover reflects the intended title and duration before clicking Save.
