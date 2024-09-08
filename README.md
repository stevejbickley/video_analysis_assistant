<<<<<<< HEAD
### Video Analysis Assistant ###

A project that utilizes OpenAI's Chat Completions API to develop the "Video Analysis Assistant" a specialized virtual assistant designed to interpret and categorize online/television ads according to the "Cognitive Bias Codex" by Gust de Backer (Source: https://gustdebacker.com/cognitive-biases/). 
=======
### Blursday Thematic Coding Bot/Assistant ###

A project that utilizes OpenAI's Assistant API to develop the "Blursday Thematic Coding Assistant," a specialized virtual assistant designed to interpret and categorize responses according to the "Blursday Codebook" developed by  

See key references: 
1. https://osf.io/359qm/
2. https://app.gorilla.sc/openmaterials/278377
3. https://dnacombo.shinyapps.io/Blursday/

This assistant is designed to analyze and code data on individuals' perceptions of time and temporal planning. It classifies how people recall past events and plan for future activities within a week, month, and year, providing insights into how individuals perceive and manage time.
>>>>>>> main

## Preparation
 
# Navigate to the folder

<<<<<<< HEAD
=======
```cd ./stevejbickley/video_analysis_assistant```

OR

>>>>>>> main
```cd ./video_analysis_assistant```

# Clone the project and run the following commands:

```poetry env use path_to_pyevn_python_version```

If you are not using pyenv, just replace the above command with:

```poetry env use path_to_python_interpreter```

Activate the virtual environment if needed:

```source ./.venv/bin/activate```

If using windows - to activate environment, run the following:
<<<<<<< HEAD

```poetry shell```
=======
```cd ./video_analysis_assistant```
```poetry shell```
```cd ../ ```
>>>>>>> main

# Next, run the following to update poetry and install ffmpeg (if required):

```poetry update```

Macos
```brew install ffmpeg```

Linux
```apt install ffmpeg```

Running Python and Scripts:

```poetry run python script.py```

<<<<<<< HEAD
 Note: ffmpeg is open-source suite of libraries and programs for handling video, audio, and other multimedia files and streams
=======
# Note: ffmpeg is open-source suite of libraries and programs for handling video, audio, and other multimedia files and streams
>>>>>>> main


## Creating your own poetry environment

<<<<<<< HEAD
# Initialise the root/current folder (if required):
=======
# To create new poetry project (if required):

```poetry new video_analysis_assistant```


# Initialize the existing directory (if required):

```cd video_analysis_assistant``` 

followed by:
>>>>>>> main

```poetry init```


# To add a new package to your project, use:

```poetry add package-name```



