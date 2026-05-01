# Archive old PDFs from Downloads in Finder

In macOS Finder, cut every PDF in the Downloads folder that is older than a configured age and paste the files into the Documents Archive folder.

## Parameters

- `file_age_days` (integer, optional, default: 7) — age threshold in days; files older than this are moved.
- `source_folder` (string, required, default: "~/Downloads") — folder scanned for PDFs to archive.
- `target_folder` (string, required, default: "~/Documents/Archive") — destination folder that receives the archived PDFs.

## Preconditions

- Finder has read and write access to both {source_folder} and {target_folder}.
- The {target_folder} directory already exists (this workflow does not create it).

## Steps

1. Switch focus to Finder and open the {source_folder} window.
2. Sort the file list by Date Modified (descending) and select every PDF older than {file_age_days} days.
3. Press Command+X to cut the selected PDFs.
4. Navigate Finder to the {target_folder} window.
5. ⚠️ [DESTRUCTIVE] Press Command+V to transfer the cut PDFs from {source_folder} into {target_folder}.

## Expected outcome

Every PDF in {source_folder} older than {file_age_days} days is now located in {target_folder}, and the originals are no longer present in {source_folder}.

## Notes

The paste step is flagged destructive because moving files between Finder folders cannot be reverted with Command+Z once Finder has committed the operation. The Runner should confirm the selection count before step 5.
