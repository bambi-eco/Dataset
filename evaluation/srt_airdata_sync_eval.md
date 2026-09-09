# Is the SRT + AirData combination of the pose extractor necessary?

`bambi_detection`'s `timed_pose_extractor.py` takes the drone position of every
video frame from the AirData flight log (precise GPS, 5 or 10 Hz) and uses the
DJI SRT subtitle file (one entry per frame, coarser GPS) for three things: the
timestamp of every frame, one clock offset between the two logs (fitted by
matching the two GPS traces), and positions inside AirData gaps longer than
2 s. The question was whether interpolating the AirData rows alone would do,
with the guess that straight transects are fine and corners might not be.

**Answer: the SRT is needed, but not for the corners.** Linear interpolation
between AirData rows is accurate to centimetres at the native 5 to 10 Hz and
stays below 0.5 m even at 1 Hz in the sharpest turns. What the SRT buys is the
time synchronisation: without it the only handle on the video's start is the
AirData `isVideo` flag, and that flag comes on 0.0 to 1.0 s (once 4.7 s) after
the first frame, a different amount on every flight. At 3 to 5 m/s that is
0.5 to 4 m of position error on every frame, straight legs included, and on
straight legs it is worst because that is where the drone is fastest. Two
flights of the sample also show the cases where AirData alone cannot work at
all: a 322 s hole in the log inside the video, and a flight made of five
separate recordings.

## Method

`srt_airdata_sync_eval.py` needs only the non-media files of the raw release
(`download_from_zenodo.py --version raw --annotations-only`): the thermal SRT
files and `air_data.csv`. For each flight it

1. reproduces the extractor: AirData rows of the `isVideo` run, SRT frame
   times, the clock offset fitted by least squares on the two GPS traces, then
   the AirData position interpolated at every frame time. For flights 146 and
   14 this reproduction matches the published `_matched_poses.json` to 0.00 m
   and 0 ms, so everything below is measured against the extractor's own
   output;
2. classifies the flight by the course-over-ground rate of the AirData track:
   straight (< 5 deg/s), turning (> 20 deg/s), both only while moving faster
   than 1 m/s;
3. measures the alternatives:
   - **[A] no SRT**: frame times from the `isVideo` onset plus a constant
     30 fps (and, as a variant, the true mean frame rate), positions from
     AirData;
   - **[B] lower log rate**: every k-th AirData row kept, the dropped rows
     interpolated linearly in time and compared with their true values;
   - **[C] SRT GPS only**: the SRT positions after the same clock offset;
   - **[D] heading**: compass heading interpolated between rows at a lower
     rate, and SRT gimbal yaw against AirData gimbal heading.

## Flights 146 (Mavic 3T, 5 Hz AirData) and 14 (Matrice 30T, 10 Hz)

```
====================================================================================================
flight 146: 2 SRT file(s), 13646 video frames, AirData 5 Hz, video segment 457 s (2282 rows), gaps > 2 s: 0, largest gap 0.40 s
  median ground speed while moving 4.90 m/s, straight 74 % / turning 0 % of the video time
  fitted SRT->AirData clock offset +0.206 s   (first frame on the AirData clock: 11:39:00.905990)
  reproduction vs published poses: time |dt| median 0 ms, position n= 12191  mean   0.00  median   0.00  p95   0.00  max   0.01

[A] AirData only, no SRT: frame times from the isVideo flag onset + constant 30 fps
    isVideo onset is +494 ms from the fitted first-frame time; over the video the SRT frame clock drifts +282 ms against 30 fps
    position error, all frames (m)             n= 13646  mean   1.22  median   1.24  p95   2.37  max   2.86
      straight flight                          n= 10104  mean   1.60  median   1.67  p95   2.42  max   2.86
      turning                                  n=     2  mean   0.78  median   0.78  p95   0.80  max   0.80
    same with the true frame rate (29.981 fps) n= 13646  mean   1.70  median   2.33  p95   2.67  max   3.01
    a pure timing error of dt costs speed x dt: at the median speed 4.9 m/s -> 0.1 s = 0.49 m, 0.5 s = 2.45 m

[B] AirData only, correctly synced, but at a lower log rate (linear interpolation between rows)
    error of the interpolated rows against the rows that were dropped, metres
     1.7 Hz (0.6 s between rows): all          n=  1521  mean   0.03  median   0.01  p95   0.11  max   0.14
          straight                             n=  1123  mean   0.03  median   0.01  p95   0.12  max   0.14
          turning                                (no samples)
     1.0 Hz (1.0 s between rows): all          n=  1825  mean   0.07  median   0.02  p95   0.26  max   0.40
          straight                             n=  1350  mean   0.08  median   0.02  p95   0.28  max   0.40
          turning                              n=     1  mean   0.21  median   0.21  p95   0.21  max   0.21
     0.5 Hz (2.0 s between rows): all          n=  2053  mean   0.24  median   0.06  p95   0.90  max   1.44
          straight                             n=  1515  mean   0.24  median   0.04  p95   0.92  max   1.44
          turning                              n=     1  mean   0.98  median   0.98  p95   0.98  max   0.98
     0.2 Hz (5.0 s between rows): all          n=  2190  mean   1.25  median   0.72  p95   4.09  max   5.36
          straight                             n=  1617  mean   1.10  median   0.33  p95   4.13  max   5.36
          turning                              n=     1  mean   2.59  median   2.59  p95   2.59  max   2.59

[C] SRT GPS only (no AirData), after the same clock offset
    position error (m), all frames             n= 13646  mean   0.37  median   0.37  p95   0.65  max   1.01
      straight                                 n= 10104  mean   0.38  median   0.37  p95   0.67  max   1.01
      turning                                  n=     2  mean   0.25  median   0.25  p95   0.26  max   0.26
    SRT latitude/longitude have 5 decimals -> 1.11 m quantisation

[D] heading between AirData rows: compass heading interpolated at a lower log rate
     1.7 Hz: heading error (deg), all          n=  1521  mean   0.05  median   0.03  p95   0.20  max   1.17
          straight                             n=  1123  mean   0.06  median   0.03  p95   0.20  max   1.17
          turning                                (no samples)
     1.0 Hz: heading error (deg), all          n=  1825  mean   0.07  median   0.02  p95   0.26  max   1.50
          straight                             n=  1350  mean   0.07  median   0.02  p95   0.26  max   1.50
          turning                              n=     1  mean   0.08  median   0.08  p95   0.08  max   0.08
     0.5 Hz: heading error (deg), all          n=  2053  mean   0.08  median   0.03  p95   0.30  max   1.80
          straight                             n=  1515  mean   0.08  median   0.03  p95   0.31  max   1.80
          turning                              n=     1  mean   0.22  median   0.22  p95   0.22  max   0.22
    SRT gimbal yaw vs AirData gimbal heading (deg) n= 13646  mean   3.70  median   3.70  p95   3.77  max   4.16
====================================================================================================
flight 14: 2 SRT file(s), 18393 video frames, AirData 10 Hz, video segment 615 s (5926 rows), gaps > 2 s: 0, largest gap 0.70 s
  median ground speed while moving 4.97 m/s, straight 82 % / turning 12 % of the video time
  fitted SRT->AirData clock offset +0.682 s   (first frame on the AirData clock: 08:41:12.770367)
  reproduction vs published poses: time |dt| median 0 ms, position n= 17996  mean   0.00  median   0.00  p95   0.00  max   0.02

[A] AirData only, no SRT: frame times from the isVideo flag onset + constant 30 fps
    isVideo onset is +430 ms from the fitted first-frame time; over the video the SRT frame clock drifts +636 ms against 30 fps
    position error, all frames (m)             n= 18393  mean   0.88  median   0.76  p95   2.03  max   2.43
      straight flight                          n= 15125  mean   0.92  median   0.80  p95   2.04  max   2.43
      turning                                  n=  2182  mean   0.68  median   0.57  p95   1.48  max   2.18
    same with the true frame rate (29.969 fps) n= 18393  mean   2.08  median   2.18  p95   2.26  max   2.45
    a pure timing error of dt costs speed x dt: at the median speed 5.0 m/s -> 0.1 s = 0.50 m, 0.5 s = 2.48 m

[B] AirData only, correctly synced, but at a lower log rate (linear interpolation between rows)
    error of the interpolated rows against the rows that were dropped, metres
     2.0 Hz (0.5 s between rows): all          n=  4740  mean   0.02  median   0.01  p95   0.06  max   0.45
          straight                             n=  3898  mean   0.01  median   0.01  p95   0.03  max   0.11
          turning                              n=   558  mean   0.06  median   0.05  p95   0.09  max   0.31
     1.0 Hz (1.0 s between rows): all          n=  5333  mean   0.05  median   0.02  p95   0.24  max   0.79
          straight                             n=  4386  mean   0.02  median   0.01  p95   0.08  max   0.42
          turning                              n=   628  mean   0.20  median   0.20  p95   0.34  max   0.70
     0.5 Hz (2.0 s between rows): all          n=  5629  mean   0.15  median   0.02  p95   0.86  max   2.06
          straight                             n=  4630  mean   0.05  median   0.02  p95   0.21  max   1.18
          turning                              n=   664  mean   0.71  median   0.74  p95   1.11  max   2.06
     0.2 Hz (5.0 s between rows): all          n=  5807  mean   0.75  median   0.03  p95   4.27  max   6.95
          straight                             n=  4778  mean   0.26  median   0.03  p95   1.86  max   5.83
          turning                              n=   685  mean   3.35  median   3.59  p95   5.40  max   6.95

[C] SRT GPS only (no AirData), after the same clock offset
    position error (m), all frames             n= 18393  mean   0.13  median   0.12  p95   0.26  max   0.38
      straight                                 n= 15125  mean   0.13  median   0.13  p95   0.26  max   0.38
      turning                                  n=  2182  mean   0.10  median   0.10  p95   0.21  max   0.30
    SRT latitude/longitude have 6 decimals -> 0.11 m quantisation

[D] heading between AirData rows: compass heading interpolated at a lower log rate
     2.0 Hz: heading error (deg), all          n=  4740  mean   0.06  median   0.02  p95   0.26  max   2.40
          straight                             n=  3898  mean   0.03  median   0.02  p95   0.10  max   0.78
          turning                              n=   558  mean   0.21  median   0.10  p95   0.80  max   1.58
     1.0 Hz: heading error (deg), all          n=  5333  mean   0.11  median   0.04  p95   0.49  max   2.65
          straight                             n=  4386  mean   0.05  median   0.02  p95   0.15  max   1.41
          turning                              n=   628  mean   0.42  median   0.20  p95   1.58  max   2.65
     0.5 Hz: heading error (deg), all          n=  5629  mean   0.14  median   0.06  p95   0.57  max   2.52
          straight                             n=  4630  mean   0.07  median   0.04  p95   0.20  max   1.97
          turning                              n=   664  mean   0.51  median   0.32  p95   1.84  max   2.52
    SRT gimbal yaw vs AirData gimbal heading (deg) n= 18393  mean  11.01  median  11.00  p95  11.10  max  12.70
```

Reading the numbers:

- **Interpolation between rows is not the problem.** At the native rate the
  interpolation error is a few centimetres. Even with only one row per second
  the error is 2 cm on straight legs and 0.2 m (p95 0.34 m) in the sharpest
  turns of flight 14; it takes one row per 5 s before turns lose metres. Heading
  interpolates equally well: under 0.5 deg mean in turns at 0.5 Hz.
- **The synchronisation is the problem.** The `isVideo` flag comes on 0.49 s
  (146) and 0.43 s (14) after the fitted first-frame time. On its own that
  shifts every frame by half a second along the track: 1.2 m and 0.9 m mean
  error, 2.4 m at the start. Using the true mean frame rate instead of 30 fps
  makes it worse (2.1 m mean), because on these flights the 30 fps rounding
  happened to cancel part of the onset error over the video. Turning frames
  show smaller errors than straight ones only because the drone slows down in
  the turns.
- **SRT GPS alone is usable but coarser**: 0.37 m mean on flight 146 (5
  decimal places, 1.1 m quantisation) and 0.13 m on flight 14 (6 decimals).
- SRT gimbal yaw and AirData gimbal heading differ by a constant 3.7 deg (146)
  and 11.0 deg (14); worth knowing when either is used as the camera heading.

![flight 14](srt_airdata_sync_flight14.png)

Left: the sharpest turn of flight 14. The 1 Hz interpolation (red dashes) cuts
the corner by a few decimetres; the no-SRT variant (orange) follows the same
path but every frame sits 0.4 s further along it. Right: the error of the
alternatives over the whole video; grey is the course rate, so the turns are
the grey spikes. The no-SRT error is largest on the straight legs.

## Eleven flights across drones and years

`srt_airdata_sync_eval.py --scan <folder of raw flights>`:

```
flight frames  AirData gap>2s maxgap SRT dec clock off onset err true fps noSRT mean noSRT p95 1Hz turn p95 SRT-only mean speed
     0  11566      5 Hz      0   1.6s       6    +1.20s    +0.89s   29.992      2.24m     2.66m        0.32m         0.12m  3.0
   100  11736      5 Hz      0   0.2s       6    +1.43s    +0.24s   29.984      0.46m     0.78m        0.29m         0.08m  3.0
   122  13041      5 Hz      0   0.4s       6    +0.22s    +0.78s   29.982      1.91m     2.34m        0.30m         0.09m  3.0
   146  13646      5 Hz      0   0.4s       5    +0.21s    +0.49s   29.981      1.22m     2.37m        0.21m         0.37m  4.9
   152  27270      5 Hz      0   0.2s       6    +1.17s    +0.04s   29.981      0.60m     1.32m        0.44m         0.07m  2.8
   163  12870      5 Hz      0   0.4s       6    +0.57s    +0.96s   29.975      3.75m     4.56m        0.34m         0.14m  4.8
    17  30464      5 Hz      0   1.4s       5    +0.39s    -0.40s   28.407      2.13m     5.51m        0.33m        27.40m  3.9
   277  11177     10 Hz      0   0.1s       6    +0.44s    +0.33s   29.987      0.81m     1.58m        0.34m         0.09m  3.0
   350  25196     10 Hz      1 321.8s       6    -4.11s    +4.72s   29.968      9.00m    13.42m        2.08m        82.49m  2.9
   380  11820     10 Hz      0   0.1s       6    +1.07s    +0.29s   29.990      0.44m     0.57m        0.25m         0.06m  2.0
    50  10412      5 Hz      0   1.0s       5    +0.78s    +0.38s   29.982      1.03m     1.93m         nanm         0.39m  4.7
```

Columns: AirData log rate, gaps over 2 s and the largest gap, SRT coordinate
decimals, the fitted SRT-to-AirData clock offset, the `isVideo` onset relative
to the fitted first frame, the mean frame rate implied by the SRT timestamps,
the mean and p95 position error of the no-SRT variant [A], the p95
interpolation error in turns at 1 Hz [B], the mean error of SRT GPS only [C],
and the median ground speed.

- The onset error ranges from +0.04 s to +0.96 s across the nine ordinary
  flights, so it is not a constant that could be calibrated away: without the
  SRT the sync is uncertain by about half a second, which at these speeds is
  0.4 to 3.8 m mean error per flight.
- The 1 Hz turn error is 0.2 to 0.44 m everywhere. At the native 5 to 10 Hz it
  is negligible.
- **Flight 350** has a 322 s hole in the AirData log inside the video (the log
  simply stops for five minutes and resumes). About 39 % of its frames have no
  AirData position; the extractor fills them from the SRT. The `-4.1 s` offset
  and the large errors in that row come from the reference interpolating
  across the hole, not from the SRT.
- **Flight 17** consists of five recordings started and stopped by the pilot,
  spanning 1072 s against an 807 s `isVideo` run, so neither a single onset nor
  a constant frame rate describes it. Its no-SRT and SRT-only numbers are not
  meaningful, but the case itself is the point: without per-frame timestamps
  there is no way to place the frames of such a flight.
- Flights 200, 250 and 300 could not be fetched (their raw depositions answer
  403).

## Recommendation

Keep the SRT. Its GPS positions are not needed (AirData interpolation is
better, at any of the log rates seen), but its per-frame timestamps are the
only reliable link between video time and the flight log, and the extractor's
fitted clock offset is what turns a half-second-uncertain sync into a
centimetre-level one. The gap fill from SRT positions is rare but decisive when
it happens (flight 350).

If the goal is to simplify, the part that could go is the SRT *position*
pathway: the offset fit needs the SRT GPS, but after that the SRT could be
reduced to timestamps, with AirData providing every position and the SRT
positions used only as the fallback inside log gaps. That is essentially what
the extractor already does.
