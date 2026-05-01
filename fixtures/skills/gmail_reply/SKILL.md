# Reply to a Gmail thread in Chrome

Open the Gmail web client, locate the most recent unread message from a specific sender, compose a short reply, and send it.

## Parameters

- `recipient_name` (string, required) — display name or email of the sender to reply to.
- `reply_body` (string, required) — body text of the reply.
- `reply_subject_prefix` (string, optional, default: "Re:") — prefix used by Gmail when replying.

## Preconditions

- Google Chrome is installed and signed into the target Gmail account.
- An unread message from {recipient_name} exists in the inbox.

## Steps

1. Switch focus to Google Chrome and open the Gmail inbox tab.
2. Locate and open the most recent unread thread from {recipient_name}; confirm the subject line begins with {reply_subject_prefix} once the reply pane opens.
3. Click the Reply button at the bottom of the open thread to open the compose pane.
4. Type the reply body into the compose pane: {reply_body}
5. ⚠️ [DESTRUCTIVE] Click the Send button to send the reply to {recipient_name}.

## Expected outcome

The reply thread shows the new message at the top with a "Sent just now" timestamp, and the original thread is no longer marked unread in the inbox.

## Notes

The Send action is irreversible once Gmail's undo-send window (typically 5–30 seconds, depending on account settings) elapses. The Runner should confirm before executing step 5.
