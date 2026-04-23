# Set a Slack focus status with expiry

In the Slack desktop app, open the profile menu, choose "Set a status", enter a status line and duration, and post the status to your profile.

## Parameters

- `status_text` (string, required, default: ":dart: heads down") — the status line shown to teammates, Slack emoji syntax allowed.
- `status_duration_hours` (integer, required, default: 2) — how long the status remains before Slack auto-clears it.

## Preconditions

- The Slack desktop app is installed and signed into the target workspace.
- The keyboard input language is English (so status text entry matches the configured shortcuts).

## Steps

1. Switch focus to the Slack desktop app and open the target workspace.
2. Click your avatar in the top-right corner to open the profile menu.
3. Choose "Set a status" from the profile menu to open the status editor.
4. Type the status line into the editor: {status_text}
5. Select an expiry of {status_duration_hours} hours from the "Clear after" dropdown.
6. ⚠️ [DESTRUCTIVE] Click the Save button to post the status {status_text} to your Slack profile.

## Expected outcome

Your Slack avatar shows the new status {status_text}, and teammates see the status with a timer indicating {status_duration_hours} hours remaining.

## Notes

Posting a status is visible to every teammate in the workspace, so it is flagged destructive — the Runner should confirm the status text and duration with the user before step 6.
