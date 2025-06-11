# Changelog

Actual version: "0.0.0"

## 2025-06

* [added] Several variants of suggestions for each level

## 2025-01

* [added] Expert heatmap
* [added] Tabs for different stitches

## 2024-11

* [added] Add selected reports to collection

## 2024-10

* [added] AI based movement analysis
* [added] Issue reporting

## 2024-08

* [added] If no knot start is annotated it is calculated by the stitch start and stitch end
* [changed] Scene crop is turned off

## 2024-07

* [added] relation to between static and dynamic analysis of the stitch created
* [changed] re-worked passing of tracking information into deeper functions
* [added] relative position of instruments

## 2024-06

* [added] Create stitch start and stitch end annotation buttons
* [added] Stream videos
* [added] Add mediafile to collection
* [fixed] Status message contains last line of the log on error
* [added] Velocity std, count of high velocities in output

## 2024-05

* [added] Video rotation
* [fixed] More stable stitch split detection by fixing 1D-smoothing  for small number of labels
* [fixed] Student email assignment

## 2024-03

* [added] Update spreadsheet for collection
* [added] Categories added
* [added] Show annotator and file owner for authenticated users
* [fixed] Fixed not showing some star type annotations in the report
* [added] Add into collection from the list of reports

## 2024-02

* [changed] Video is now stretched to 100% if possible
* [changed] The report can be reviewed without the processing is finished
* [added] New metrics in the review
* [added] Upload progress bar
* [added] Delete media file
* [added] Consent added
* [added] Collections of uploaded files can be run together

## 2024-01

* [added] Processing status of the mediafile
* [added] Review of the mediafile
* [added] Ask user for a review after the mediafile is uploaded and in email
* [changed] Tracking is now more continuous and not so jumpy
* [changed] Every annotation change is now stored to spreadsheet
* [added] Every stitch video part is now trimmed by presence of needle holder in the operating area
* [added] Bar plot in report with metric

* [added] Bulk import files by copying into server "drop_dir" folder
* [changed] Empty frames used as another axis in stitch split detection
* [added] Logs in webapp for superuser

## 2023-12

* [added] Menu for logged users
* [added] Hand tracking
* [added] Microsurgery support
* [added] Access to the admin page of object from web report

## 2022-11

* [added] Instrument duration in percent
* [added] Download original file button
* [added] Individual user view with list of all his reports


## 0.0


* [added] Scale of the output video is changed automatically

* [added] Ruler inserted into image
* [added] Pixel size estimation based on incision size
* [changed] Video has lower resolution to be loaded faster
* [changed] Video and graph are concatenated horizontaly

* [added] Poster to see first frame of the video before the video is loaded
* [added] Wider images
* [added] Web report