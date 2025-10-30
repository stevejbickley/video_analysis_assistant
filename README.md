### GenAI Media Mapping for Behavioural Comms Insights ###

We built a multimodal GenAI pipeline to analyse TV/online ads from Suncorp and competitors, converting raw video into structured evidence about what’s shown/said, how it’s framed, and which cognitive biases are invoked. 

The workflow produced time-aligned audio logs, frame-level video logs, and a bias/heuristic scoring table (0-10 with reasoning/justification text) for each ad, enabling comparative mapping across brands and creative variants, and informing practical recommendations for message design and optimisation.

## Our Approach

* Data capture: Collected campaign/stimulus videos; standardised formats; extracted audio and frames at 1s cadence with robust fallbacks.
* Transcription & timing: High accuracy transcription (segment-level timestamps) to align speech with visuals for later fusion.
* Event logs (multimodal): Prompted LLMs on synced frames + transcript to generate JSON event logs (scene, actors, devices/nudges, tone/cues, CTAs) per timestamp.
* Bias/heuristic evaluation: Applied a defined catalogue + JSON schema to score presence/intensity (0-10) with model-provided reasoning; exported *_biases.csv for analysis.
* Downstream analyses on outputs: Cross-brand/creative comparisons, bias & framing maps, accessibility/coverage diagnostics; datasets designed to feed dashboards, online experiments (phase 2), and SurveyLM simulations (phase 3).

## Outcomes
* Standardised, reproducible datasets per ad: (*_audioLog.csv, *_videoLog.csv, _biases.csv) ready for statistical analysis and cross-brand comparisons.
* Documented, end-to-end methodology (prompts, schemas, error handling) enabling reliability, repeat runs, and scale-out.
* Comparative bias & framing profiles by brand/product/creative, supporting practical refactoring guidance.
* Defined next steps for evidence-building: turnkey online experiments to quantify effects; agent-based simulations to test strategy in complex contexts.

## Research Team
Dr Steve Bickley, Dr Ho Fai Chan, Patricia Galliford & Prof. Benno Torgler

# Preparation
 
## Navigate to the folder

```cd ./stevejbickley/video_analysis_assistant```

OR

```cd ./video_analysis_assistant```

## Clone the project and run the following commands:

```poetry env use path_to_pyevn_python_version```

If you are not using pyenv, just replace the above command with:

```poetry env use path_to_python_interpreter```

Activate the virtual environment if needed:

```source ./.venv/bin/activate```

If using windows - to activate environment, run the following:
```cd ./video_analysis_assistant```
```poetry shell```
```cd ../ ```

## Next, run the following to update poetry and install ffmpeg (if required):

```poetry update```

Macos
```brew install ffmpeg```

Linux
```apt install ffmpeg```

Running Python and Scripts:

```poetry run python script.py```

### Note: ffmpeg is open-source suite of libraries and programs for handling video, audio, and other multimedia files and streams


# Creating your own poetry environment

## To create new poetry project (if required):

```poetry new video_analysis_assistant```


## Initialize the existing directory (if required):

```cd video_analysis_assistant``` 

followed by:

```poetry init```


## To add a new package to your project, use:

```poetry add package-name```



