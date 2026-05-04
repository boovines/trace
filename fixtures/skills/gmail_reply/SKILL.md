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

1. Switch focus to Google Chrome with the Gmail inbox tab visible (the inbox list takes up the centre column; screenshot 1).
2. Click the most recent unread thread from {recipient_name} in the inbox list — unread rows are bold and sit near the top of the centre column; confirm the subject line begins with {reply_subject_prefix} once the thread opens (screenshot 3).
3. Click the **Reply** button below the message body, just above the small reply-style icons row at the bottom of the open thread, to open the compose pane (screenshot 4).
4. Type the reply body into the compose pane that appears at the bottom of the thread: {reply_body} (screenshot 5).
5. ⚠️ [DESTRUCTIVE] Click the blue **Send** button at the bottom-left of the compose pane to send the reply to {recipient_name} (screenshot 6).

## Expected outcome

The reply thread shows the new message at the top with a "Sent just now" timestamp, and the original thread is no longer marked unread in the inbox.

## Notes

The Send action is irreversible once Gmail's undo-send window (typically 5–30 seconds, depending on account settings) elapses. The Runner should confirm before executing step 5.
