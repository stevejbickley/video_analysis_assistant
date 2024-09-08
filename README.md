### Video Analysis Assistant ###

A project that utilizes OpenAI's Chat Completions API to develop the "Video Analysis Assistant" a specialized virtual assistant designed to interpret and categorize online/television ads according to the "Cognitive Bias Codex" by Gust de Backer (Source: https://gustdebacker.com/cognitive-biases/). 

## Preparation
 
# Navigate to the folder

```cd ./video_analysis_assistant```

# Clone the project and run the following commands:

```poetry env use path_to_pyevn_python_version```

If you are not using pyenv, just replace the above command with:

```poetry env use path_to_python_interpreter```

Activate the virtual environment if needed:

```source ./.venv/bin/activate```

If using windows - to activate environment, run the following:

```poetry shell```

# Next, run the following to update poetry and install ffmpeg (if required):

```poetry update```

Macos
```brew install ffmpeg```

Linux
```apt install ffmpeg```

Running Python and Scripts:

```poetry run python script.py```

 Note: ffmpeg is open-source suite of libraries and programs for handling video, audio, and other multimedia files and streams


## Creating your own poetry environment

# Initialise the root/current folder (if required):

```poetry init```


# To add a new package to your project, use:

```poetry add package-name```



