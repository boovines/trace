# Finder Organize

Move every PDF in ~/Downloads older than N days into ~/Documents/Archive.

## Parameters

- `age_days` (number) — Minimum age of files to move, in days.
- `archive_name` (string) — Destination folder name under ~/Documents.

## Steps

1. Open Finder and navigate to the Downloads folder.
2. Switch the view to List view.
3. Click the Kind column header to sort by file type.
4. Select every PDF older than {age_days} days.
5. ⚠️ Drag the selection into the ~/Documents/{archive_name} folder.
