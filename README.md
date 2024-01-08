# Hume Many Faces Hack

## Problem
After analyzing a video, sometimes you may have additional face IDs but only have x number of known faces in your video, which can be an issue if you want to associate the appropriate Face ID and relate emotion estimates to the correct person.

This script takes a Hume output with face IDs (e.g., `face_x`) and detects all unique face IDs and groups similar face IDs, which are more than likely related to one of the unique face IDs. So, for example, if you know you have 15 faces in a video but have an output with 17 face IDs. The result after using this script will look like:

```
{'unique_face_0': {'face_0'},
 'unique_face_1': {'face_1'},
 'unique_face_10': {'face_6'},
 'unique_face_11': {'face_4'},
 'unique_face_12': {'face_7'},
 'unique_face_13': {'face_2'},
 'unique_face_14': {'face_8'},
 'unique_face_2': {'face_16'},
 'unique_face_3': {'face_10'},
 'unique_face_4': {'face_15', 'face_9', 'face_3'},
 'unique_face_5': {'face_11'},
 'unique_face_6': {'face_12'},
 'unique_face_7': {'face_5'},
 'unique_face_8': {'face_13'},
 'unique_face_9': {'face_14'}}
```

## Additional Notes
- Set the `face_config` [option](https://dev.hume.ai/docs/python-sdk-model-configs) using Hume's Python SDK, for example, to `identify_face=True`).  
- Read more about the [many faces issue](https://dev.hume.ai/docs/too-many-face-identifiers). 